import os
import uuid
from flask import Flask, render_template, request
import pandas as pd

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# -----------------------------
# Config
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"csv"}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def save_uploaded_file(file_storage):
    """Save uploaded file and return saved filepath."""
    if not file_storage or file_storage.filename.strip() == "":
        raise ValueError("No file selected.")

    if not allowed_file(file_storage.filename):
        raise ValueError("Only .csv files are allowed.")

    unique_name = f"{uuid.uuid4().hex}.csv"
    save_path = os.path.join(UPLOAD_FOLDER, unique_name)
    file_storage.save(save_path)
    return save_path

# -----------------------------
# Home
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")

# =========================================================
# Q3(a) Apriori - Association Rule Mining
# =========================================================
@app.route("/apriori", methods=["GET"])
def apriori_page():
    return render_template("apriori.html")

@app.route("/apriori/result", methods=["POST"])
def apriori_result():
    try:
        file = request.files.get("dataset")
        min_support = float(request.form.get("min_support", 0.3))
        min_confidence = float(request.form.get("min_confidence", 0.6))

        path = save_uploaded_file(file)

        # Read CSV (transactions)
        # Expected: each row is transaction, items separated by comma in columns
        df = pd.read_csv(path, header=None)

        # Convert each row to list of items (remove NaN/empty)
        transactions = []
        for row in df.values.tolist():
            items = [str(x).strip() for x in row if pd.notna(x) and str(x).strip() != ""]
            if items:
                transactions.append(items)

        if len(transactions) == 0:
            raise ValueError("CSV has no valid transactions. Please upload correct transaction data.")

        # Transaction Encoding
        te = TransactionEncoder()
        te_array = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_array, columns=te.columns_)

        # Apriori Frequent Itemsets
        frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)

        if frequent_itemsets.empty:
            raise ValueError("No frequent itemsets found. Try lowering min_support.")

        # Association Rules
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

        if rules.empty:
            raise ValueError("No rules generated. Try lowering min_confidence or min_support.")

        # Clean columns for display
        rules_display = rules.copy()
        rules_display["antecedents"] = rules_display["antecedents"].apply(lambda x: ", ".join(list(x)))
        rules_display["consequents"] = rules_display["consequents"].apply(lambda x: ", ".join(list(x)))

        # Select important columns
        show_cols = ["antecedents", "consequents", "support", "confidence", "lift"]
        rules_display = rules_display[show_cols].sort_values(by="confidence", ascending=False)

        return render_template(
            "apriori_result.html",
            min_support=min_support,
            min_confidence=min_confidence,
            total_transactions=len(transactions),
            total_items=len(df_encoded.columns),
            table_html=rules_display.to_html(index=False, classes="table")
        )

    except Exception as e:
        return render_template("error.html", title="Apriori Error", message=str(e))

# =========================================================
#Q3(b) DBSCAN - Clustering
# =========================================================
@app.route("/dbscan", methods=["GET"])
def dbscan_page():
    return render_template("dbscan.html")

@app.route("/dbscan/result", methods=["POST"])
def dbscan_result():
    try:
        file = request.files.get("dataset")
        eps = float(request.form.get("eps", 0.8))
        min_samples = int(request.form.get("min_samples", 3))

        path = save_uploaded_file(file)

        df = pd.read_csv(path)

        if df.empty:
            raise ValueError("CSV is empty.")

        # Keep only numeric columns
        numeric_df = df.select_dtypes(include=["number"])

        if numeric_df.shape[1] == 0:
            raise ValueError("DBSCAN requires numeric columns. Your CSV has no numeric columns.")

        # Scale
        scaler = StandardScaler()
        X = scaler.fit_transform(numeric_df.values)

        # DBSCAN
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)

        result_df = df.copy()
        result_df["Cluster"] = labels

        # Summary
        unique_labels = sorted(set(labels))
        noise_count = int((labels == -1).sum())
        cluster_count = len([l for l in unique_labels if l != -1])

        return render_template(
            "dbscan_result.html",
            eps=eps,
            min_samples=min_samples,
            used_numeric_columns=list(numeric_df.columns),
            cluster_count=cluster_count,
            noise_count=noise_count,
            table_html=result_df.to_html(index=False, classes="table")
        )

    except Exception as e:
        return render_template("error.html", title="DBSCAN Error", message=str(e))

# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
