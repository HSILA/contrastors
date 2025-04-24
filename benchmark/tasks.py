from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata
from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class ChemBenchRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ChemBenchRetrieval",
        dataset={
            "path": "BASF-AI/ChemBenchRetrieval",
            "revision": "ed5d61c2e6374149e248d9703035d2ae8e272df0",
        },
        description="A dataset from one-hop questions of ChemMultiHop",
        reference="https://github.com/HSILA/ChemMultiHop",
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
    )


class ChemRxivNC1(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ChemRxivNC1",
        dataset={
            "path": "BASF-AI/ChemRxivNC1",
            "revision": "ddab7adead7ed3b971543ab5b2679ca54cc3eb99",
        },
        description="ChemRxiv paragraphs NC, with 500 querues and 5500 corpus",
        reference="https://github.com/HSILA/ChemMultiHop",
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
    )
