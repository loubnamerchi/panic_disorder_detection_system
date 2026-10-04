import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")


def _save(fig: plt.Figure, name: str, save_dir: Optional[str]) -> None:
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        path = Path(save_dir) / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")


class EDAVisualizer:
    """Produce EDA and preprocessing visualizations."""

    def __init__(self, save_dir: Optional[str] = "reports/figures") -> None:
        self.save_dir = save_dir

    def plot_overview_table(
        self,
        df: pd.DataFrame,
        title: str = "Dataset Overview"
    ) -> plt.Figure:

        rows = [
            ["Shape", f"{df.shape[0]:,} rows × {df.shape[1]} columns"],
            ["Memory", f"{df.memory_usage(deep=True).sum() / 1e6:.2f} MB"],
            ["Numeric cols", str(df.select_dtypes(include=np.number).shape[1])],
            ["Categorical cols",
                str(df.select_dtypes(include=["object", "category"]).shape[1])
            ],
            ["Missing cells", f"{df.isnull().sum().sum():,}"],
            ["Duplicate rows", f"{df.duplicated().sum():,}"],
        ]

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.axis("off")

        table = ax.table(
            cellText=rows,
            colLabels=["Metric", "Value"],
            cellLoc="left",
            loc="center",
            bbox=[0, 0, 1, 1],
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)

        fig.suptitle(
            title,
            fontsize=13,
            fontweight="bold"
        )

        _save(fig, "01_overview_table", self.save_dir)

        return fig
    
    def plot_target_distribution(
        self,
            df: pd.DataFrame,
            column: str,
            title: str | None = None,
            bins: int = 30
        ) -> plt.Figure:
    
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.set_style("white")
        sns.histplot(
            data=df,
            x=column,
            bins=bins,
            kde=True,
            ax=ax,
            color="steelblue",
        )
    
        ax.set_title(
            title or f"{column} Distribution",
            fontsize=13,
            fontweight="bold"
        )
    
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
    
        fig.tight_layout()
    
        _save(fig, f"02_histogram_target_distribution{column}", self.save_dir)
    
        return fig
    
    def plot_age_distribution(self,
        df: pd.DataFrame,
        title: str | None = None,
        bins: int = 30,
    ) -> plt.Figure:
    
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.set_style("white")
        sns.histplot(
            data=df,
            x="Age",
            bins=bins,
            kde=True,
            ax=ax,
            color="steelblue",
        )
    
        ax.set_title(
            title or "Age Distribution",
            fontsize=13,
            fontweight="bold",
        )
    
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
    
        fig.tight_layout()
    
        _save(
            fig,
            "12_histogram_age_distribution",
            self.save_dir,
        )
    
        return fig
    
    

        
    def plot_missing_values_heatmap(
        self,
            df: pd.DataFrame,
            title: str = "Missing Values Heatmap"
        ) -> plt.Figure:
    
        fig, ax = plt.subplots(figsize=(10, 5))
    
        sns.heatmap(
            df.isnull(),
            cbar=False,
            yticklabels=False,
            ax=ax
        )
    
        ax.set_title(
            title,
            fontsize=13,
            fontweight="bold"
        )
    
        ax.set_xlabel("Columns")
        ax.set_ylabel("Rows")
    
        fig.tight_layout()
    
        _save(fig, "03_missing_values_heatmap", self.save_dir)
    
        return fig
    
    
    def plot_numerical_distributions(
        self,
        df: pd.DataFrame,
        target: str,
        columns: list[str] | None = None,
        bins: int = 30
    ) -> plt.Figure:
    
        if columns is None:
            columns = df.select_dtypes(include="number").columns.drop(target).tolist()
    
        n_cols = 3
        n_rows = int(np.ceil(len(columns) / n_cols))
    
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(15, 4 * n_rows)
        )
    
        axes = np.array(axes).reshape(-1)
    
        for ax, col in zip(axes, columns):
            ax.hist(df[col].dropna(), bins=bins)
            ax.set_title(col)
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            ax.grid(alpha=0.2)
    
        for ax in axes[len(columns):]:
            ax.axis("off")
    
        fig.suptitle(
            "Numerical Variable Distributions",
            fontsize=16,
            fontweight="bold"
        )
    
        fig.tight_layout()
        
        _save(fig, "04_histogram_numerical_distributions", self.save_dir)
    
        return fig
    
    
    def plot_categorical_distributions(self,
            df: pd.DataFrame,
            columns: list[str] | None = None,
            top_n: int | None = None
        ) -> plt.Figure:
    
        if columns is None:
            columns = df.select_dtypes(
                include=["object", "category", "bool"]
            ).columns.drop(["Participant ID"], errors="ignore").tolist()
        
        
        n_cols = 3
        n_rows = int(np.ceil(len(columns) / n_cols))
    
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(15, 4 * n_rows)
        )
    
        axes = np.array(axes).reshape(-1)
    
        for ax, col in zip(axes, columns):
    
            counts = df[col].value_counts(dropna=False)
    
            if top_n is not None:
                counts = counts.head(top_n)
    
            counts.plot(
                kind="bar",
                ax=ax
            )
    
            ax.set_title(col)
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=45)
            ax.grid(axis="y", alpha=0.2)
    
        for ax in axes[len(columns):]:
            ax.axis("off")
    
        fig.suptitle("Categorical Variable Distributions",fontsize=16,fontweight="bold")
    
        fig.tight_layout()
    
        _save(fig,"05_barplot_categorical_distributions",self.save_dir )
    
        return fig
    
    
    def plot_unique_values(self,
            df: pd.DataFrame,
            columns: list[str] | None = None
        ) -> plt.Figure:
    
        if columns is None:
            columns = df.columns.tolist()
    
        unique_counts = df[columns].nunique()
    
        fig, ax = plt.subplots(figsize=(12, 6))
    
        unique_counts.sort_values(ascending=False).plot(
            kind="bar",
            ax=ax
        )
    
        # Add number of unique values above each bar
        for i, value in enumerate(unique_counts.sort_values(ascending=False)):
            ax.text(
                i,
                value,
                str(value),
                ha="center",
                va="bottom"
            )
    
        ax.set_title(
            "Number of Unique Values per Variable",
            fontsize=16,
            fontweight="bold"
        )
    
        ax.set_xlabel("Variable")
        ax.set_ylabel("Number of Unique Values")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.2)
    
        fig.tight_layout()
    
        _save(
            fig,
            "06_unique_values_barchart",
            self.save_dir
        )
    
        return fig
        
    
    def plot_numerical_features_vs_target(self,
            df: pd.DataFrame,
            target: str,
            columns: list[str] | None = None
        ) -> plt.Figure:
    
        # Select numerical columns automatically
        if columns is None:
            columns = df.select_dtypes(include="number").columns.tolist()
    
            # Exclude target
            if target in columns:
                columns.remove(target)
    
        # Create subplots
        n_cols = 2
        n_rows = int(np.ceil(len(columns) / n_cols))
    
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(12, 5 * n_rows)
        )
    
        # Make axes iterable
        axes = np.array(axes).reshape(-1)
    
        # Plot each numerical feature against target
        for ax, column in zip(axes, columns):
    
            ax.scatter(
                df[column],
                df[target],
                alpha=0.6
            )
    
            ax.set_title(
                f"{column} vs {target}",
                fontsize=14,
                fontweight="bold"
            )
    
            ax.set_xlabel(column)
            ax.set_ylabel(target)
    
            ax.grid(
                alpha=0.2
            )
    
        # Hide unused axes
        for ax in axes[len(columns):]:
            ax.set_visible(False)
    
        fig.suptitle(
            f"Numerical Features vs {target}",
            fontsize=16,
            fontweight="bold"
        )
    
        fig.tight_layout()
    
        _save(
            fig,
            "08_numerical_features_vs_target",
            self.save_dir
        )
    
        return fig
    
    
    def plot_correlation_heatmap(self,
            df: pd.DataFrame,
            columns: list[str] | None = None
        ) -> plt.Figure:
    
        # Select numerical columns automatically
        if columns is None:
            columns = df.select_dtypes(
                include="number"
            ).columns.tolist()
    
        # Calculate correlation matrix
        correlation = df[columns].corr()
    
        # Create figure
        fig, ax = plt.subplots(
            figsize=(12, 8)
        )
    
        # Plot heatmap
        sns.heatmap(
            correlation,
            ax=ax,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            square=True
        )
    
        ax.set_title(
            "Correlation Heatmap",
            fontsize=16,
            fontweight="bold"
        )
    
        fig.tight_layout()
    
        _save(
            fig,
            "10_correlation_heatmap",
            self.save_dir
        )
    
        return fig
    
    
    def plot_categorical_relationships(self,
            df: pd.DataFrame,
            columns: list[str] | None = None
        ) -> plt.Figure:
    
        # Select categorical columns automatically
        if columns is None:
            columns = df.select_dtypes(
                include=["object", "category", "bool"]
            ).columns.tolist()
    
        # Check that there are at least two categorical columns
        if len(columns) < 2:
            raise ValueError(
                "At least two categorical columns are required."
            )
    
        # Create all pairs
        pairs = []
    
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                pairs.append(
                    (columns[i], columns[j])
                )
    
        # Create subplots
        n_cols = 2
        n_rows = int(np.ceil(len(pairs) / n_cols))
    
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(14, 5 * n_rows)
        )
    
        # Make axes iterable
        axes = np.array(axes).reshape(-1)
    
        # Plot each categorical relationship
        for ax, (column1, column2) in zip(
            axes,
            pairs
        ):
    
            sns.countplot(
                data=df,
                x=column1,
                hue=column2,
                ax=ax
            )
    
            ax.set_title(
                f"{column1} vs {column2}",
                fontsize=14,
                fontweight="bold"
            )
    
            ax.set_xlabel(column1)
            ax.set_ylabel("Count")
    
            ax.tick_params(
                axis="x",
                rotation=45
            )
    
            ax.grid(
                axis="y",
                alpha=0.2
            )
    
        # Hide unused axes
        for ax in axes[len(pairs):]:
            ax.set_visible(False)
    
        fig.suptitle(
            "Categorical Features Relationships",
            fontsize=16,
            fontweight="bold"
        )
    
        fig.tight_layout()
    
        _save(
            fig,
            "11_categorical_features_relationships",
            self.save_dir
        )
    
        return fig
    
    
    def plot_boxplots(self,
            df: pd.DataFrame,
            target: str | None = None,
            columns: list[str] | None = None
        ) -> plt.Figure:
    
        # Select numerical columns automatically
        if columns is None:
            columns = df.select_dtypes(include="number").columns.tolist()
    
            # Exclude target if provided
            if target is not None and target in columns:
                columns.remove(target)
    
        # Create one subplot per feature
        n_cols = 2
        n_rows = int(np.ceil(len(columns) / n_cols))
    
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(14, 6 )
        )
    
        # Make axes iterable
        axes = np.array(axes).reshape(-1)
    
        for ax, col in zip(axes, columns):
    
            sns.boxplot(
                data=df,
                x=col,
                ax=ax,
                color="lightcoral"
            )
    
            ax.set_title(
                f"Boxplot of {col}",
                fontsize=14
            )
    
            ax.set_xlabel(col)
    
            ax.grid(
                axis="y",
                alpha=0.2
            )
    
        # Hide unused subplots
        for ax in axes[len(columns):]:
            ax.axis("off")
    
        fig.suptitle(
            "Boxplots of Continuous Variables",
            fontsize=16,
            fontweight="bold"
        )
    
        fig.tight_layout(
            rect=[0, 0, 1, 0.95]
        )
    
        _save(
            fig,
            "07_boxplots_numerical_variables",
            self.save_dir
        )
    
        return fig
    
    
    def plot_categorical_features_vs_target(self,
        df: pd.DataFrame,
        target: str,
        columns: list[str] | None = None
    ) -> plt.Figure:
    
        # Automatically select categorical features
        if columns is None:
            columns = (
                df.select_dtypes(
                    include=["object", "category", "bool"]
                )
                .columns
                .drop([target], errors="ignore")
                .tolist()
            )
    
        # Remove identifier if present
        columns = [
            col for col in columns
            if col != "Participant ID"
        ]
    
        # Create subplot grid
        n_cols = 2
        n_rows = int(np.ceil(len(columns) / n_cols))
    
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(16, 5 * n_rows)
        )
    
        axes = np.array(axes).reshape(-1)
    
        # Plot each categorical feature
        for ax, col in zip(axes, columns):
    
            # Calculate percentages within each category
            proportions = pd.crosstab(
                df[col],
                df[target],
                normalize="index"
            ) * 100
    
            # Stacked bar plot
            proportions.plot(
                kind="bar",
                stacked=True,
                ax=ax
            )
    
            ax.set_title(
                f"{col} vs {target}",
                fontsize=13,
                fontweight="bold"
            )
    
            ax.set_xlabel(col)
            ax.set_ylabel("Percentage (%)")
    
            ax.legend(
                title=target,
                labels=[
                    "No Panic Disorder",
                    "Panic Disorder"
                ]
            )
    
            ax.tick_params(
                axis="x",
                rotation=45
            )
    
            ax.grid(
                axis="y",
                alpha=0.2
            )
    
        # Hide unused subplots
        for ax in axes[len(columns):]:
            ax.axis("off")
    
        fig.suptitle(
            "Categorical Features vs Panic Disorder Diagnosis",
            fontsize=17,
            fontweight="bold"
        )
    
        fig.tight_layout(
            rect=[0, 0, 1, 0.96]
        )
    
        _save(
            fig,
            "09_categorical_features_vs_target",
            self.save_dir
        )
    
        return fig
            
    def plot_target_rate_by_feature(self,
        df: pd.DataFrame,
        target: str,
        columns: list[str] | None = None
    ) -> None:
    
        # Automatically select categorical features
        if columns is None:
            columns = df.select_dtypes(include="object").columns.tolist()
    
        # Remove target and identifier if present
        columns = [
            col for col in columns
            if col != target and col != "Participant ID"
        ]
    
        for col in columns:
    
            # Calculate target rate for each category
            rates = (
                df.groupby(col)[target]
                .mean()
                .mul(100)
                .sort_values()
            )
    
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 5))
    
            rates.plot(
                kind="barh",
                ax=ax
            )
    
            ax.set_xlabel("Panic Disorder Rate (%)")
            ax.set_ylabel(col)
            ax.set_title(
                f"Panic Disorder Rate by {col}",
                fontsize=14,
                fontweight="bold"
            )
    
            ax.grid(
                axis="x",
                alpha=0.2
            )
    
            fig.tight_layout()
    
            # Save each figure
            _save(
                fig,
                f"11_Target_rate_by_feature_{col}",
                self.save_dir
            )
    
        return fig
    
    def plot_distribution_by_target(self,
        df: pd.DataFrame,
        feature: str,
        target: str,
        title: str | None = None,
    ) -> plt.Figure:
    
        fig, ax = plt.subplots(figsize=(8, 5))
    
        sns.boxplot(
            data=df,
            x=target,
            y=feature,
            ax=ax,
        )
    
        ax.set_title(
            title or f"{feature} Distribution by {target}",
            fontsize=13,
            fontweight="bold",
        )
    
        ax.set_xlabel(target)
        ax.set_ylabel(feature)
    
        fig.tight_layout()
    
        _save(
            fig,
            f"13_boxplot_{feature.lower().replace(' ', '_')}_by_target",
            self.save_dir,
        )
    
        return fig
            
    
    def plot_categorical_by_target(self,
        df: pd.DataFrame,
        feature: str,
        target: str,
        title: str | None = None,
    ) -> plt.Figure:
    
        fig, ax = plt.subplots(figsize=(8, 5))
    
        sns.countplot(
            data=df,
            x=feature,
            hue=target,
            ax=ax,
        )
    
        ax.set_title(
            title or f"{feature} Distribution by {target}",
            fontsize=13,
            fontweight="bold",
        )
    
        ax.set_xlabel(feature)
        ax.set_ylabel("Count")
    
        fig.tight_layout()
    
        _save(
            fig,
            f"14_countplot_{feature.lower().replace(' ', '_')}_by_target",
            self.save_dir,
        )
    
        return fig
    
    
    def plot_model_comparison_table(self,
        comparison_results: dict,
        models: list[str],
        title: str = "Model Comparison",
    ) -> plt.Figure:
    
        rows = []
    
        for model_name in models:
            rows.append({
                "Model": model_name,
    
                "Val PR-AUC": comparison_results[
                    model_name
                ]["validation"]["pr_auc"],
    
                "Val ROC-AUC": comparison_results[
                    model_name
                ]["validation"]["roc_auc"],
    
                "Val F1": comparison_results[
                    model_name
                ]["validation"]["f1"],
    
                "Val Recall": comparison_results[
                    model_name
                ]["validation"]["recall"],
    
                "Test PR-AUC": comparison_results[
                    model_name
                ]["test"]["pr_auc"],
    
                "Test ROC-AUC": comparison_results[
                    model_name
                ]["test"]["roc_auc"],
    
                "Test F1": comparison_results[
                    model_name
                ]["test"]["f1"],
    
                "Test Recall": comparison_results[
                    model_name
                ]["test"]["recall"],
            })
    
        comparison_df = pd.DataFrame(rows)
    
        fig, ax = plt.subplots(
            figsize=(14, 5)
        )
    
        ax.axis("off")
    
        ax.set_title(
            title,
            fontsize=13,
            fontweight="bold",
            pad=20,
        )
    
        table = ax.table(
            cellText=comparison_df.round(4).values,
            colLabels=comparison_df.columns,
            loc="center",
            cellLoc="center",
        )
    
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)
    
        fig.tight_layout()
    
        _save(
            fig,
            "15_model_comparison",
            self.save_dir,
        )
    
        return fig
    
    
        
        
    
    
        
        
        
        
        
        
        

        