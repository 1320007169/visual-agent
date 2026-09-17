"""Mixed relation and free-form perception samples with the same tool prompt."""

from verl.utils.dataset.zwz_original_relation_dataset import ZwzOriginalRelationDataset


class ZwzDeepEyesV2Dataset(ZwzOriginalRelationDataset):
    def __getitem__(self, item):
        source = self.dataframe[item]
        source_name = source.get("data_source")
        if source_name not in {"visual-agent-zwz-relation", "visual-agent-deepeyesv2", "visual-agent-hrbench4k"}:
            raise ValueError(f"Unexpected mixed dataset source: {source_name!r}")
        row = super().__getitem__(item)
        row["extra_info"]["split"] = source.get("split", "train")
        if source_name != "visual-agent-zwz-relation":
            answer = str(source["solution"]).strip()
            row["data_source"] = source_name
            row["ability"] = source["ability"]
            row["reward_model"] = {"style": "rule", "ground_truth": answer}
            row["extra_info"] = {
                "source": source_name, "data_source": source_name,
                "question": source["question"], "answer": answer,
                "split": source.get("split", "train"), "index": item,
                "source_image": source["source_image"],
            }
        return row
