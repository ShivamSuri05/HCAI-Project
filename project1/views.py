from django.http import HttpResponse, JsonResponse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
from io import BytesIO
import base64
from django.shortcuts import render
from .forms import UploadCSVForm
from django.core.files.storage import FileSystemStorage
from .utils import handle_missing, handle_outliers, evaluate_classification, evaluate_regression
from django.views.decorators.csrf import csrf_exempt
from django.template.loader import render_to_string
from django.conf import settings
import os
import uuid
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score


def upload_csv(request):
    result = None
    error = None
    context = {}

    if request.method == 'POST':
        action = request.POST.get('action') 

        form = UploadCSVForm(request.POST, request.FILES)
        if action == 'upload':
            if form.is_valid():
                try:
                    csv_file = request.FILES['file']
                
                    temp_path = os.path.join(settings.MEDIA_ROOT, 'temp_uploaded.csv')

                    with open(temp_path, 'wb+') as destination:
                        for chunk in csv_file.chunks():
                            destination.write(chunk)

                    df = pd.read_csv(temp_path)
    
                    request.session['temp_csv_path'] = temp_path

                    # removing duplicate entries
                    df = df.drop_duplicates()

                    head = df.head().to_html(classes="table", index=False)
                    tail = df.tail().to_html(classes="table", index=False)

                    columns = df.columns.tolist()

                    # Missing values
                    df.replace('', pd.NA, inplace=True)
                    missing_data = df.isnull().sum()
                    missing_data = missing_data[missing_data > 0].to_dict()

                    # Outlier detection using IQR
                    outliers = {}
                    for col in df.select_dtypes(include=['float64', 'int64']).columns:
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        outlier_rows = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
                        if not outlier_rows.empty:
                            outliers[col] = len(outlier_rows)

                    context.update({
                        'form': form,
                        'result': result,
                        'error': error,
                        'head': head,
                        'tail': tail,
                        'missing_data': missing_data,
                        'outliers': outliers,
                        'columns': columns
                    })

                except Exception as e:
                    error = f"Error processing file: {e}"
                    print(error)
            else:
                error = "Invalid form submission."
    else:
        form = UploadCSVForm()
        context['form'] = form
        context['error'] = error

    #context['form'] = form
    #context['error'] = error
    return render(request, 'project1/upload.html', context)

def generate_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    numeric_df = df.select_dtypes(include=['number'])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Heatmap")
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

@csrf_exempt
def generate_plots(request):
    if request.method == 'POST':
        try:
            temp_path = request.session.get('temp_csv_path')

            if not temp_path:
                return JsonResponse({'error': 'No CSV file found'}, status=400)
            
            df = pd.read_csv(temp_path)  # reload saved CSV
            target_col = request.POST.get('target_column')
            request.session["target_column"] = target_col

            if not target_col or target_col not in df.columns:
                return JsonResponse({'error': 'Invalid or missing target column'}, status=400)

            # Handle missing values
            for col in df.columns:
                missing_key = f'missing_{col}'
                if missing_key in request.POST:
                    strategy = request.POST[missing_key]
                    df = handle_missing(df, col, strategy)

            # Handle outliers only for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                outlier_key = f'outlier_{col}'
                if outlier_key in request.POST:
                    strategy = request.POST[outlier_key]
                    df = handle_outliers(df, col, strategy)

            # Save cleaned df to a new temp file
            cleaned_filename = f"cleaned_{uuid.uuid4().hex}.csv"
            cleaned_path = os.path.join(settings.MEDIA_ROOT, cleaned_filename)  # Use your actual temp directory path
            df.to_csv(cleaned_path, index=False)

            # Update session to point to cleaned file
            request.session['temp_csv_path'] = cleaned_path

            plots_html = ""

            for col in df.columns:
                if col == target_col:
                    continue  # Skip the target itself

                fig, ax = plt.subplots(figsize=(6,4))

                # Plotting differently based on data type of the column and target
                if pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(df[target_col]):
                    # Scatter plot for numeric vs numeric
                    ax.scatter(df[col], df[target_col], alpha=0.6)
                    ax.set_xlabel(col)
                    ax.set_ylabel(target_col)
                    ax.set_title(f'{col} vs {target_col}')
                
                elif pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_numeric_dtype(df[target_col]):
                    # Boxplot numeric col grouped by categorical target
                    df.boxplot(column=col, by=target_col, ax=ax)
                    plt.suptitle('')
                    ax.set_title(f'{col} by {target_col}')
                    ax.set_xlabel(target_col)
                    ax.set_ylabel(col)

                elif not pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(df[target_col]):
                    # Boxplot target by categorical feature
                    df.boxplot(column=target_col, by=col, ax=ax)
                    plt.suptitle('')
                    ax.set_title(f'{target_col} by {col}')
                    ax.set_xlabel(col)
                    ax.set_ylabel(target_col)

                else:
                    # Both categorical: maybe countplot or stacked bar plot
                    cross_tab = pd.crosstab(df[col], df[target_col])
                    cross_tab.plot(kind='bar', stacked=True, ax=ax)
                    ax.set_title(f'{col} vs {target_col}')
                    ax.set_xlabel(col)
                    ax.set_ylabel('Count')
                    plt.xticks(rotation=45)

                buf = BytesIO()
                plt.tight_layout()
                plt.savefig(buf, format='png')
                plt.close(fig)
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')

                # Appending to html string
                plots_html += f'<div style="margin-bottom: 20px;"><img src="data:image/png;base64,{img_base64}" alt="{col} vs {target_col}"/></div>'

            corr_img = generate_correlation_heatmap(df)
            plots_html += f'<div style="margin-bottom: 20px;"><img src="data:image/png;base64,{corr_img}" alt="Correlation Heatmap"/></div>'
            
            train_form_html = render_to_string("project1/train_model_form.html", {"columns": df.columns})
            return JsonResponse({
                'html': plots_html,
                "train_html": train_form_html
                })

        except Exception as e:
            print(e)
            return JsonResponse({'error': str(e)}, status=500)

def index(request):
    return HttpResponse("Welcome to Project 1!")

@csrf_exempt
def train_model(request):
    if request.method == 'POST':
        try:
            print(request.POST.dict())
            temp_path = request.session.get('temp_csv_path')
            if not temp_path:
                return JsonResponse({'error': 'CSV not found'}, status=400)

            df = pd.read_csv(temp_path)
            target = target = request.session.get('target_column')
            model_type = request.POST.get('model_type')
            
            split_ratio = float(request.POST.get('test_size'))
            
            random_state = int(request.POST.get('random_state', 42))

            # For Linear Regression
            fit_intercept = request.POST.get('fit_intercept') == 'on'  # bool (checkbox)

            # For Logistic Regression
            regularization_type = request.POST.get('regularization_type', 'l2')  # str
            regularization_strength = float(request.POST.get('regularization_strength', 1.0))  # float

            # For Random Forest
            n_estimators = int(request.POST.get('n_estimators', 100))  # int, default 100
            max_depth = request.POST.get('max_depth')
            max_depth = int(max_depth) if max_depth else None  # int or None
            min_samples_split = int(request.POST.get('min_samples_split', 2))  # int
            min_samples_leaf = int(request.POST.get('min_samples_leaf', 1))  # int

            if target not in df.columns:
                return JsonResponse({'error': 'Invalid target column'}, status=400)

            X = df.drop(columns=[target])
            X = pd.get_dummies(X)
            y = df[target]
            # Encode target if classification model is selected
            if model_type in ['logistic_regression', 'random_forest']:
                if y.dtype == 'object' or y.dtype.name == 'category':
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    y = le.fit_transform(y)

            if model_type in ['linear_regression']:
                if y.dtype == 'object' or y.dtype.name == 'category':
                    # Convert categorical y to numeric codes (single column)
                    y = y.astype('category').cat.codes


            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=random_state)

            if model_type == 'linear_regression':
                model = LinearRegression(fit_intercept=fit_intercept)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                evaluation_metrics = evaluate_regression(y_test, y_pred)

            elif model_type == 'logistic_regression':
                # Map regularization_type to solver and penalty properly:
                penalty = regularization_type
                solver = 'lbfgs'  # default solver; for l1 or elasticnet, solver must be 'saga'
                
                if penalty == 'l1' or penalty == 'elasticnet':
                    solver = 'saga'
                if penalty == 'none':
                    penalty = 'none'

                model = LogisticRegression(
                    penalty=penalty,
                    C=regularization_strength,
                    max_iter=1000,
                    solver=solver,
                    l1_ratio=0.5 if penalty == 'elasticnet' else None,  # l1_ratio needed for elasticnet
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                evaluation_metrics = evaluate_classification(y_test, y_pred)

            elif model_type == 'random_forest':
                print(f"RandomForest parameters: n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}")
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=random_state,
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                evaluation_metrics = evaluate_classification(y_test, y_pred)

            else:
                return JsonResponse({'error': 'Invalid model type'}, status=400)

            return JsonResponse({
                'message': 'Model trained successfully', 
                'metrics': evaluation_metrics
            })

        except Exception as e:
            print(e)
            return JsonResponse({'error': str(e)}, status=500)