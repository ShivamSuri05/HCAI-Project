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
import json
import pickle
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def index(request):
    context = request.session.get('decision_context', {})

    if request.method == "POST":
        if 'model_type' in request.POST and request.POST['model_type'] == 'sparse':
            context.update(sparse_tree(request))
        elif 'model_type' in request.POST and request.POST['model_type'] == 'lr':
            context.update(logistic_regression())
        elif 'model_type' in request.POST and request.POST['model_type'] == 'lr_sparse':
            context.update(sparse_logistic_regression(request))
        elif 'model_type' in request.POST and request.POST['model_type'] == 'counterfactual':
            context.update(counterfactual_examples(request))
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    model_path = os.path.join(BASE_DIR, 'project3/models/decision_tree.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    num_leaves = clf.get_n_leaves()

    fig, ax = plt.subplots(figsize=(16, 9))
    plot_tree(clf, feature_names=X.columns, class_names=clf.classes_, filled=True, ax=ax)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    image_uri = 'data:image/png;base64,' + image_base64

    sampled_indices = np.random.choice(X_test.index, size=30, replace=False)
    sampled_rows = X_test.loc[sampled_indices].to_dict(orient='records')
    sampled_labels = y_test.loc[sampled_indices].tolist()

    examples = []
    for idx, features, label in zip(sampled_indices, sampled_rows, sampled_labels):
        features_clean = {k: (v.item() if hasattr(v, 'item') else v) for k, v in features.items()}
        label_clean = label.item() if hasattr(label, 'item') else label
        examples.append({'index': int(idx), 'features': features_clean, 'true_label': label_clean})

    context = {
        'accuracy': f"{accuracy:.4f}",
        'num_leaves': int(num_leaves),
        'tree_image': image_uri,
        'gosdt_error': None,
        'examples': examples,
        'class_labels': np.unique(y).tolist()
    }

    return context


def build_tree(dot, node_dict, node_id=0, parent_id=None, edge_label=""):
    current_id = str(node_id)

    if node_dict.get("name") == "class":
        label = f"Predict: {node_dict['prediction']}\nLoss: {node_dict['loss']:.4f}\nComplexity: {node_dict['complexity']:.4f}"
        dot.node(str(node_id), label=label, shape="box", style="filled")
    else:
        label = f"{node_dict['name']} {node_dict['relation']} {node_dict['reference']}"
        dot.node(current_id, label, style="filled")

        if "true" in node_dict:
            node_id += 1
            node_id = build_tree(dot, node_dict["true"], node_id, current_id, "True")

        if "false" in node_dict:
            node_id += 1
            node_id = build_tree(dot, node_dict["false"], node_id, current_id, "False")

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

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

    try:
        lambda_float = float(lambda_value)
        C_value = 1.0 / lambda_float if lambda_float != 0 else 1e6  # Avoid division by zero
    except ValueError:
        C_value = 1.0

    
    model = LogisticRegression(penalty='l1', solver='saga', C=C_value, max_iter=10000)
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(model.n_iter_[0])

    used_features = np.count_nonzero(np.any(model.coef_ != 0, axis=0))

    context = {
        'sp_lr_accuracy': f"{accuracy:.4f}",
        'sp_lr_num_features_used': int(used_features),
        'sp_lr_lambda_value': f"{lambda_float:.3f}",
        'sp_lr_iterations': int(model.n_iter_[0])
    }

    return context

def counterfactual_examples(request):
    X, y = load_and_clean_dataset()
    model_path = os.path.join(BASE_DIR, 'project3/models/decision_tree.pkl')
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)

    x_index = int(request.POST.get('example_index'))
    target_label = request.POST.get('target_label')
    top_k = int(request.POST.get('topk', 5))
    top_k = min(top_k, 10)

    x = X.loc[x_index].values.reshape(1, -1)

    categorical_cols = ['island', 'sex', 'year']
    numerical_cols = [col for col in X.columns if col not in categorical_cols]
    cat_idx = [X.columns.get_loc(c) for c in categorical_cols]
    num_idx = [X.columns.get_loc(c) for c in numerical_cols]

    N = 5000
    rng = np.random.default_rng(42)
    std_devs = X[numerical_cols].std().values + 1e-6
    noise = rng.normal(0, std_devs * 0.3, size=(N, len(numerical_cols)))
    
    x_numerical = x[0, num_idx]
    x_samples = np.repeat(x, N, axis=0)
    x_samples[:, num_idx] += noise

    for idx, col in zip(cat_idx, categorical_cols):
        unique_vals = X[col].unique()
        x_samples[:, idx] = rng.choice(unique_vals, size=N)

    preds = clf.predict(x_samples)
    matching = x_samples[preds == target_label]

    ex_list = X.loc[x_index].values.tolist()
    example = f"#{x_index} - Features: island={int(ex_list[0])}, bill_length_mm={ex_list[1]}, \
    bill_depth_mm={ex_list[2]}, flipper_length_mm={ex_list[3]}, body_mass_g={ex_list[4]}, \
    sex={int(ex_list[5])}, year={int(ex_list[6])}"

    if len(matching) == 0:
        return {
            'counterfactuals': [],
            'feature_names': list(X.columns),
            'message': f"No counterfactuals found for this example: {example}"
        }

    mad = np.median(np.abs(X[numerical_cols] - X[numerical_cols].median()), axis=0) + 1e-6
    distances = hybrid_distance(x[0], matching, mad, num_idx, cat_idx)
    top_k_indices = np.argsort(distances)[:top_k]
    top_k_samples = matching[top_k_indices]

    top_k_samples_df = pd.DataFrame(top_k_samples, columns=X.columns)

    list_of_lists = top_k_samples_df.values.tolist()
    for row in list_of_lists:
        for idx, col in enumerate(top_k_samples_df.columns):
            if col in categorical_cols:
                row[idx] = int(row[idx])
            else:
                row[idx] = round(row[idx],1)
    

    return {
        'counterfactuals': list_of_lists,
        'feature_names': list(X.columns),
        'true_label': y.loc[x_index],
        'selected_example': example,
        'target_label': target_label
    }

def hybrid_distance(x, X_cf, mad, num_idx, cat_idx):
    num_diff = np.abs(X_cf[:, num_idx] - x[num_idx]) / mad
    num_dist = np.sum(num_diff, axis=1)

    cat_diff = (X_cf[:, cat_idx] != x[cat_idx]).astype(int)
    cat_dist = np.sum(cat_diff, axis=1)

    return num_dist + cat_dist
