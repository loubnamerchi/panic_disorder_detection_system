# LEAKAGE / SYNTHETIC-RULE AUDIT

# Investigation 1:Category-level target rates + deterministic-pattern detection
# Investigation 2: Statistical association testing implemented in statistical_tests.py

import pandas as pd
from src.analysis.statistical_tests import StatisticalTests
from src.utils.logger import get_logger
from src.utils.config import load_config



logger = get_logger(__name__)


class LeakageAudit:


    def __init__(self, config_path: str = "config.yaml") -> None:
        self.cfg = load_config(config_path)
        self.data_cfg = self.cfg["data"]
        self.target = self.data_cfg["target_column"]

        logger.info(
            "LeakageAudit initialized. Target column: '%s'",
            self.target,
        )

    def get_categorical_columns(
        self,
        df: pd.DataFrame,
    ) -> list[str]:

        categorical_columns = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        if self.target in categorical_columns:
            categorical_columns.remove(self.target)

        logger.debug(
            "Identified %d categorical predictors.",
            len(categorical_columns),
        )

        return categorical_columns


    def audit_category_patterns(
        self,
        df: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:

        categorical_columns = self.get_categorical_columns(df)

        results: dict[str, pd.DataFrame] = {}

        logger.info(
            "Starting Investigation 1: "
            "category-level target rate audit."
        )

        for column in categorical_columns:

            audit = (
                df.groupby(
                    column,
                    dropna=False,
                )[self.target]
                .agg(
                    Count="count",
                    Positive_Count="sum",
                    Positive_Rate="mean",
                )
                .reset_index()
            )

            # Convert proportion to percentage
            audit["Positive_Rate"] *= 100

            audit = audit.sort_values(
                "Positive_Rate",
                ascending=False,
            ).reset_index(drop=True)

            # Flag deterministic categories
            audit["Deterministic"] = (
                audit["Positive_Rate"].eq(0)
                | audit["Positive_Rate"].eq(100)
            )

            results[column] = audit

        logger.info(
            "Investigation 1 completed for %d categorical predictors.",
            len(results),
        )

        return results


    def get_deterministic_summary(
        self,
        results: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:

        summary_rows = []

        for feature, audit in results.items():

            deterministic = audit[
                audit["Deterministic"]
            ].copy()

            if deterministic.empty:
                continue

            for _, row in deterministic.iterrows():

                summary_rows.append(
                    {
                        "Feature": feature,
                        "Category": row[feature],
                        "Count": row["Count"],
                        "Positive_Count": row["Positive_Count"],
                        "Positive_Rate": row["Positive_Rate"],
                    }
                )

        if not summary_rows:

            return pd.DataFrame(
                columns=[
                    "Feature",
                    "Category",
                    "Count",
                    "Positive_Count",
                    "Positive_Rate",
                ]
            )

        return pd.DataFrame(summary_rows).reset_index(drop=True)


    def run_audit(self,df: pd.DataFrame,) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        logger.info("LEAKAGE / SYNTHETIC-RULE AUDIT" )
        logger.info("Investigation 1: " "Category-level target rates and deterministic patterns.")

        category_results = self.audit_category_patterns(df)
        deterministic_summary =self.get_deterministic_summary(category_results)

        logger.info("Investigation 2: ""Statistical association testing ""(Chi-square + Cramér's V).")
        
        stat_tests = StatisticalTests(self.cfg)
        chi_results = stat_tests.categorical_statistical_tests(df)
        age_results = stat_tests.age_statistical_association(df)


        logger.info("LEAKAGE AUDIT COMPLETE")

        return deterministic_summary, chi_results,age_results