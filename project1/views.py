from django.http import HttpResponse, JsonResponse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
from io import BytesIO
import base64
from django.shortcuts import render
from .forms import UploadCSVForm
from django.core.files.storage import FileSystemStorage
from .utils import handle_missing, handle_outliers
from django.views.decorators.csrf import csrf_exempt
from django.template.loader import render_to_string
from django.conf import settings
import os
import seaborn as sns


def upload_csv(request):
    result = None
    error = None
    context = {}

    if request.method == 'POST':
        action = request.POST.get('action')
        print("User action:", action)  

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
            if not target_col or target_col not in df.columns:
                return JsonResponse({'error': 'Invalid or missing target column'}, status=400)

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
            return JsonResponse({'html': plots_html})

        except Exception as e:
            print("error")
            print(e)
            return JsonResponse({'error': str(e)}, status=500)

def index(request):
    return HttpResponse("Welcome to Project 1!")