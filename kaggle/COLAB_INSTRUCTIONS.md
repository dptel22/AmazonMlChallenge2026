# Run the BER pipeline in Google Colab

1. Copy the repository to Google Drive, preserving its folder structure. The
   Drive project folder must contain `student_resource/dataset/{train,test}/`
   and `code/business_entity_resolution/{src,requirements.txt,tests/}`.
2. Open `kaggle/colab_pipeline.ipynb` in Colab. Select a **High-RAM CPU**
   runtime, then run the cells in order. In the Drive setup cell, edit
   `PROJECT_ROOT` if your folder is not
   `/content/drive/MyDrive/AmazonMlChallenge2026`.
3. Stage B measures cosine recall on 20,000 validation entities. For the
   configured proxy, it selects the smallest tested `TOPK` under the default
   caps that reaches the 95% recall floor. It also prints estimated full-train
   candidate volume. If no tested K passes, it stops before candidate generation.
4. Stage E must report `Stage E PASS`; otherwise model fitting is blocked. Once
   it passes, Stage F trains LightGBM on the real features and prints held-out
   macro F_0.5 and the selected threshold.
5. Model state is checkpointed under `colab_state/`; final TSVs are under
   `colab_output/`. After a runtime disconnect, reopen the notebook and rerun
   from the beginning; completed prep/model artifacts are reused where their
   manifests permit.

The recall table is based on a validation subsample, and the full-fit F_0.5 is
the first meaningful model-quality result. The tiny local smoke test is not a
score. A valid final package still needs the measured results entered in
`student_resource/Documentation_template.md`.
