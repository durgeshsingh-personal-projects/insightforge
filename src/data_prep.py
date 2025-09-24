# src/data_prep.py

import pandas as pd

def load_data(path: str = "data/sales_data.csv") -> pd.DataFrame:
    """
    Load the sales_data.csv file with correct dtypes.
    Ensures Date is parsed as datetime and numeric values are cast properly.
    """
    df = pd.read_csv(
        path,
        parse_dates=["Date"],
        dtype={
            "Product": str,
            "Region": str,
            "Sales": float,
            "Customer_Age": int,
            "Customer_Gender": str,
            "Customer_Satisfaction": float,
        },
    )
    return df


def basic_aggregates(df: pd.DataFrame) -> dict:
    """
    Compute core aggregate statistics for sales data.
    Returns a dictionary of DataFrames and summary dicts.
    """
    res = {}
    df2 = df.copy()

    # Create a 'month' column for monthly aggregation
    df2["month"] = df2["Date"].dt.to_period("M").astype(str)

    # Sales trends over time
    res["sales_by_month"] = df2.groupby("month")["Sales"].sum().reset_index().rename(columns={"Sales": "revenue"})

    # Product performance
    res["sales_by_product"] = (
        df2.groupby("Product")["Sales"]
        .sum()
        .reset_index()
        .rename(columns={"Sales": "revenue"})
        .sort_values("revenue", ascending=False)
    )

    # Regional performance
    res["sales_by_region"] = (
        df2.groupby("Region")["Sales"]
        .sum()
        .reset_index()
        .rename(columns={"Sales": "revenue"})
        .sort_values("revenue", ascending=False)
    )

    # Customer age stats
    res["customer_age_stats"] = df2["Customer_Age"].describe().to_dict()

    # Customer satisfaction stats
    res["satisfaction_stats"] = df2["Customer_Satisfaction"].describe().to_dict()

    return res


def filter_data(
    df: pd.DataFrame,
    product: str = None,
    region: str = None,
    start_date: str = None,
    end_date: str = None,
) -> pd.DataFrame:
    """
    Filter the dataset based on product, region, and optional date range.
    Dates should be strings in YYYY-MM-DD format.
    """
    filtered = df.copy()

    if product and product in filtered["Product"].unique():
        filtered = filtered[filtered["Product"] == product]

    if region and region in filtered["Region"].unique():
        filtered = filtered[filtered["Region"] == region]

    if start_date:
        filtered = filtered[filtered["Date"] >= pd.to_datetime(start_date)]

    if end_date:
        filtered = filtered[filtered["Date"] <= pd.to_datetime(end_date)]

    return filtered


if __name__ == "__main__":
    # Quick test
    df = load_data()
    print("Dataset preview:")
    print(df.head())

    aggs = basic_aggregates(df)
    print("\nSales by month:")
    print(aggs["sales_by_month"].head())

    print("\nTop products:")
    print(aggs["sales_by_product"].head())

    print("\nRegional sales:")
    print(aggs["sales_by_region"].head())

    print("\nCustomer age stats:")
    print(aggs["customer_age_stats"])

    print("\nCustomer satisfaction stats:")
    print(aggs["satisfaction_stats"])
