from django.shortcuts import render
from django.http import HttpResponse
from palmerpenguins import load_penguins
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
from sklearn.ensemble import GradientBoostingClassifier
import sklearn.utils.validation as val
import sklearn.base
sklearn.base.check_X_y = val.check_X_y
from gosdt import GOSDT
from graphviz import Digraph
import time
from sklearn.linear_model import LogisticRegression
import numpy as np


def index(request):
    context = request.session.get('decision_context', {})

    if request.method == "POST":
        if 'model_type' in request.POST and request.POST['model_type'] == 'sparse':
            context.update(sparse_tree(request))
        elif 'model_type' in request.POST and request.POST['model_type'] == 'lr':
            context.update(logistic_regression())
        elif 'model_type' in request.POST and request.POST['model_type'] == 'lr_sparse':
            context.update(sparse_logistic_regression(request))
        else:
            context.update(decision_tree())

    request.session['decision_context'] = context
    return render(request, 'project3/index.html', context)


def load_and_clean_dataset():
    penguins = load_penguins()
    penguins.dropna(inplace=True)
    penguins['sex'] = penguins['sex'].astype('category').cat.codes
    penguins['island'] = penguins['island'].astype('category').cat.codes
    X = penguins.drop(columns=['species'])
    y = penguins['species']
    return X, y


def decision_tree():
    X, y = load_and_clean_dataset()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Model
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    num_leaves = clf.get_n_leaves()

    # Plot the tree
    fig, ax = plt.subplots(figsize=(16, 9))
    plot_tree(clf, feature_names=X.columns, class_names=clf.classes_, filled=True, ax=ax)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    image_uri = 'data:image/png;base64,' + image_base64

    context = {
        'accuracy': f"{accuracy:.4f}",
        'num_leaves': int(num_leaves),
        'tree_image': image_uri,
        'gosdt_error': None
    }

    return context


def build_tree(dot, node_dict, node_id=0, parent_id=None, edge_label=""):
    current_id = str(node_id)

    # Leaf node
    if node_dict.get("name") == "class":
        #label = f"Predict: {node_dict['prediction']}"
        label = f"Predict: {node_dict['prediction']}\nLoss: {node_dict['loss']:.4f}\nComplexity: {node_dict['complexity']:.4f}"
        dot.node(str(node_id), label=label, shape="box", style="filled")
        #dot.node(current_id, label, shape="box")
    else:
        # Internal node
        label = f"{node_dict['name']} {node_dict['relation']} {node_dict['reference']}"
        dot.node(current_id, label, style="filled")

        # True branch
        if "true" in node_dict:
            node_id += 1
            node_id = build_tree(dot, node_dict["true"], node_id, current_id, "True")

        # False branch
        if "false" in node_dict:
            node_id += 1
            node_id = build_tree(dot, node_dict["false"], node_id, current_id, "False")

    # Add edge from parent to current node
    if parent_id is not None:
        dot.edge(parent_id, current_id, label=edge_label)

    return node_id


def visualize_gosdt_repr(tree_repr_dict):
    dot = Digraph()
    build_tree(dot, tree_repr_dict)
    return dot


def sparse_tree(request):
    lambda_value = float(request.POST.get("lambda", 0.01))
    time_limit = float(request.POST.get("time_limit", 60))

    X, y = load_and_clean_dataset()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    config = {
        "regularization": lambda_value,
        "depth_budget": 10,
        "time_limit": time_limit,
        "similar_support": False,
        "verbose":False
    }

    try:
        model = GOSDT(config)  
        start = time.time()
        model1=model.fit(X_train, y_train)
        train_time = time.time() - start
    except Exception as gosdt_error:
        context = {
            'gosdt_error': f"Training failed with λ={lambda_value}. Try a smaller value. Error: {str(gosdt_error)}",
            'lambda_value': lambda_value,
            'sparse_dc_accuracy': None
        }  
        return context
        

    n_leaves = model.leaves()
    n_nodes = model.nodes()
    iterations = model.iterations

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    tree_dict = model.tree.__repr__()
    dot = visualize_gosdt_repr(tree_dict)

    graph_bytes = dot.pipe(format='png')
    image_base64 = base64.b64encode(graph_bytes).decode('utf-8')
    image_uri = 'data:image/png;base64,' + image_base64


    context = {
        "lambda_value": float(lambda_value),
        "sparse_dc_accuracy": f"{accuracy:.4f}",
        "sp_num_leaves": int(n_leaves),
        "sp_num_nodes": int(n_nodes),
        "sp_tree_image": image_uri,
        "sp_training_time": f"{train_time:.3f}",
        "sp_iterations": int(iterations),
        "gosdt_error": None
    }

    return context


def logistic_regression():
    X, y = load_and_clean_dataset()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Model (no sparsity / penalty)
    model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Number of non-zero features (used features)
    if model.coef_.ndim == 1:
        used_features = np.count_nonzero(model.coef_)
    else:
        used_features = np.count_nonzero(np.any(model.coef_ != 0, axis=0))

    context = {
        'lr_accuracy': f"{accuracy:.4f}",
        'lr_num_features_used': int(used_features)
    }

    return context


def sparse_logistic_regression(request):
    lambda_value = float(request.POST.get("lr_lambda", 0.01))
    X, y = load_and_clean_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Convert λ to C (inverse regularization strength)
    try:
        lambda_float = float(lambda_value)
        C_value = 1.0 / lambda_float if lambda_float != 0 else 1e6  # Avoid division by zero
    except ValueError:
        C_value = 1.0

    # Use L1 penalty for sparsity, saga supports L1 for multiclass
    model = LogisticRegression(penalty='l1', solver='saga', C=C_value, max_iter=10000)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(model.n_iter_[0])

    # Count how many features are used (non-zero in any class)
    used_features = np.count_nonzero(np.any(model.coef_ != 0, axis=0))

    context = {
        'sp_lr_accuracy': f"{accuracy:.4f}",
        'sp_lr_num_features_used': int(used_features),
        'sp_lr_lambda_value': f"{lambda_float:.3f}",
        'sp_lr_iterations': int(model.n_iter_[0])
    }

    return context
