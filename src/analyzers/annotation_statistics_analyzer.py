import os
import pandas as pd
from typing import Dict, Any
from collections import defaultdict, Counter

from analyzers.base import Analyzer, register_analyzer

@register_analyzer
class AnnotationStatisticsAnalyzer(Analyzer):
    def get_name(self) -> str:
        return "Annotation Statistics Analyzer"

    def get_description(self) -> str:
        return "Analyzes annotation practices, label sources, and item-related metadata."

    def analyze(self, data: Dict[str, Dict[str, Any]], output_dir: str = 'outputs') -> str:
        analyzer_dir = self.get_analyzer_output_dir(output_dir)
        output_file = os.path.join(analyzer_dir, 'annotation_statistics.txt')
        period_filter = 0

        df = pd.DataFrame(data.values())

        if period_filter != 0:
            df = df[df['period'] == period_filter]

        def count_matches(col, mapping):
            counts = Counter()
            total = 0
            for val in df[col].fillna("no information"):
                val = str(val).strip().lower()
                if val == 'not applicable' or val == 'not labelled':
                    continue
                for label, keywords in mapping.items():
                    if any(k.lower() in val.lower() for k in keywords):
                        counts[label] += 1
                        break
                else:
                    counts['no information'] += 1
                total += 1
            return counts, total

        results = []

        # Human Labels
        mapping = {
            "yes for some": ["yes for some"],
            "yes / implicit yes": ["yes", "implicit yes"],
            "no / machine": ["No / machine labelled"],
            "no information": ["no information", "unsure"]
        }
        counts, total = count_matches("Human Labels", mapping)
        results.append(("Human Labels", counts, total))

        # OG Labels (conditional)
        mapping = {
            "mixed": ["mix og, external"],
            "original": ["OG"],
            "external": ["external"],
            "no information": ["no information", "unsure"]
        }
        counts, total = count_matches("OG Labels", mapping)
        results.append(("Original Labels", counts, total))

        # Label Source
        mapping = {
            "mturk": ["turk", "amt"],
            "no information": ["no information"],
            "other": [ "" ]
        }
        counts, total = count_matches("Label Source", mapping)
        results.append(("Label Source", counts, total))

        # Labeller Population Rationale
        mapping = {
            "no information": ["no information"],
            "something": [""]
        }
        counts, total = count_matches("Labeller Population Rationale", mapping)
        results.append(("Labeller Population Rationale", counts, total))

        # Prescreening
        mapping = {
            "generic skill based": ["generic skill based"],
            "previous platform": ["previous platform"],
            "location based": ["location"],
            "no information": ["no information"],
            "other": [ "" ]
        }
        counts, total = count_matches("Prescreening", mapping)
        results.append(("Prescreening", counts, total))

        # Compensation
        mapping = {
            "no information": ["no information"],
            "money": ["money"],
            "volunteer": ["volunteer"],
            "other": [""]
        }
        counts, total = count_matches("Compensation", mapping)
        results.append(("Compensation", counts, total))

        # Total Labellers
        mapping = {
            "not mentioned": ["no information"],
            "mentioned": [""]
        }
        counts, total = count_matches("Total Labellers", mapping)
        results.append(("Total Labellers", counts, total))

        # Annotators per item
        counts, total = count_matches("Annotators per item",mapping)
        results.append(("Annotators per item", counts, total))

        # Overlap
        mapping = {
            "yes, for all": ["yes, for all"],
            "yes, for some": ["yes, for some"],
            "no information": ["no information", "unsure"],
            "no": ["no"]
        }
        counts, total = count_matches("Overlap", mapping)
        results.append(("Overlap", counts, total))

        # IRR
        irr_counts = Counter()
        overlap_column = df["Overlap"].fillna("no information").str.strip().str.lower()
        irr_column = df["IRR"].fillna("no information").str.strip().str.lower()
        total = 0
        for ov_val, irr_val in zip(overlap_column, irr_column):
            if ov_val == 'no':
                continue
            elif ov_val == 'no information':
                irr_counts['maybe not applicable'] += 1
                total += 1
            elif irr_val != 'no information' and irr_val != 'not applicable':
                irr_counts['calculated'] += 1
                total += 1
            elif irr_val == 'no information':
                irr_counts['not calculated'] += 1
                total += 1
        results.append(("IRR", irr_counts, total))

        # Training
        mapping = {
            "some training": ["some training"],
            "no information": ["no information"]
        }
        counts, total = count_matches("Training", mapping)
        results.append(("Training", counts, total))

        # Formal Instructions
        mapping = {
            "formal instructions": ["formal instructions"],
            "no information": ["no information"]
        }
        counts, total = count_matches("Formal Instructions", mapping)
        results.append(("Formal Instructions", counts, total))

        # Annotation Schema
        mapping = {
            "no information": ["no information", "no"],
            "some schema": [""]
        }
        counts, total = count_matches("a Priori Annotation Schema", mapping)
        results.append(("Annotation Schema", counts, total))

        # Annotation Schema Rationale
        schema_rationale_counts = Counter()
        schema_column = df["a Priori Annotation Schema"].fillna("no information").str.strip().str.lower()
        schema_rationale_column = df["Annotation Schema Rationale"].fillna("no information").str.strip().str.lower()
        total = 0
        for schema_val, rationale_val in zip(schema_column, schema_rationale_column):
            if schema_val == 'no information' or schema_val == 'not applicable':
                continue
            elif rationale_val != 'no information':
                schema_rationale_counts["some schema rationale"] += 1
                total += 1
            else:
                schema_rationale_counts["no information"] += 1
                total +=1 
        results.append(("Annotation Schema Rationale (if present)", schema_rationale_counts, total))

        # Item Population
        mapping = {
            "no information": ["no information"],
            "item population described": [""]
        }
        counts, total = count_matches("Item Population", mapping)
        results.append(("Item Population", counts, total))

        # Item Population Rationale
        mapping = {
            "no information": ["no information"],
            "some population rationale": [""]
        }
        counts, total = count_matches("Item Population Rationale", mapping)
        results.append(("Item Population Rationale", counts, total))
        
        # Item source
        mapping = {
            "no information": ["no information"],
            "label source given": [""]
        }
        counts, total = count_matches("Item Source", mapping)
        results.append(("Item Source", counts, total))

        # Item Sample Size Rationale
        mapping = {
            "no information": ["no information"],
            "item size rationale given": [""]
        }
        counts, total = count_matches("Item Sample Size Rationale", mapping)
        results.append(("Item Sample Size Rationale", counts, total))

        # A Priori Sample Size
        mapping = {
            "no information": ["no information"],
            "a priori sample size": [""]
        }
        counts, total = count_matches("a Priori Sample Size", mapping)
        results.append(("A Priori Sample Size", counts, total))

        # Link to Data
        mapping = {
            "yes, but broken": ["yes, but broken"],
            "yes": ["yes"],
            "no": ["no"]
        }
        counts, total = count_matches("Link to Data", mapping)
        results.append(("Link to Data", counts, total))

        # Write results
        with open(output_file, 'w') as f:
            f.write("=== Annotation Statistics ===\n\n")
            f.write(f"Selected period: {period_filter}\n\n")
            for section, counts, total in results:
                f.write(f"{section}:\n")
                for k, v in counts.items():
                    pct = (v / total) * 100 if total > 0 else 0
                    f.write(f"  {k}: {v} ({pct:.2f}%)\n")
                f.write("\n")

        return output_file
