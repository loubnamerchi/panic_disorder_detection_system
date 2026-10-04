import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from src.utils.logger import get_logger
from src.utils.config import load_config
from scipy.stats import pointbiserialr
import pandas as pd

logger = get_logger(__name__)


class StatisticalTests:

    def __init__(self, config_path: dict) -> None:
        self.cfg = config_path
        self.data_cfg = self.cfg["data"]
        self.target = self.data_cfg["target_column"]

        logger.info(
            f"StatisticalTests initialized with target: {self.target}"
        )

    def categorical_statistical_tests(self,df: pd.DataFrame,columns: list[str] | None = None) -> pd.DataFrame:

        target = self.target

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

        # Remove identifier
        columns = [
            col for col in columns
            if col != "Participant ID"
        ]

        results = []

        for col in columns:

            # Contingency table
            table = pd.crosstab(
                df[col].fillna("Missing"),
                df[target]
            )

            # Chi-square test
            chi2, p_value, dof, expected = chi2_contingency(table)

            # Cramér's V
            n = table.sum().sum()

            r, k = table.shape

            phi2 = chi2 / n

            phi2corr = max(
                0,
                phi2 - ((k - 1) * (r - 1)) / (n - 1)
            )

            rcorr = r - ((r - 1) ** 2) / (n - 1)

            kcorr = k - ((k - 1) ** 2) / (n - 1)

            denominator = min(
                kcorr - 1,
                rcorr - 1
            )

            if denominator > 0:
                cramers_v = np.sqrt(
                    phi2corr / denominator
                )
            else:
                cramers_v = np.nan

            results.append({
                "Feature": col,
                "Chi2": chi2,
                "Degrees_of_Freedom": dof,
                "p_value": p_value,
                "Cramers_V": cramers_v
            })

        # Results dataframe
        results_df = pd.DataFrame(results)

        # Sort by p-value
        results_df = (
            results_df
            .sort_values("p_value")
            .reset_index(drop=True)
        )

        logger.info(
            f"Chi-square analysis completed for {len(columns)} features."
        )

        return results_df
    
    def age_statistical_association(self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Test the association between Age and the binary target
        using point-biserial correlation.
        """
    
        target = self.target
        age = df["Age"]
        diagnosis = df[target]
    
        # Remove missing values
        valid_data = pd.concat(
            [age, diagnosis],
            axis=1
        ).dropna()
    
        r, p_value = pointbiserialr(
            valid_data["Age"],
            valid_data[target]
        )
    
        result = pd.DataFrame([{
            "Feature": "Age",
            "Correlation": r,
            "p_value": p_value,
            "Sample_Size": len(valid_data)
        }])
    
        logger.info(
            f"Age association test completed: "
            f"r={r:.4f}, p={p_value:.4e}"
        )
    
        return result
    