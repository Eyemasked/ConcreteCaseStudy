CONCRETE STRENGTH ROUND 2

1. Extract the ZIP into a folder. Both CSV inputs are included.
2. Open concrete_round2.ipynb in Jupyter and run its cells in order.
3. Results appear in the outputs folder.

Alternatively, install dependencies using: python -m pip install -r requirements.txt
Then run: python concrete_round2.py
To recompute the selected ensemble validation score: python concrete_round2.py --validate
Validation takes longer than just fitting the final models.

For Colab: upload the notebook, then upload both included CSVs. Run the setup cell and model cell. Use the final cell to download results.

Training MSE and R-squared measure fit on the 670 training observations. The recorded repeated five-fold MSE was used to select models and ensemble weights, so it is not an independent evaluation of the final selection. The true holdout MSE is unknown.

Source documentation:
https://catboost.ai/docs/en/concepts/python-reference_catboostregressor
https://scikit-learn.org/stable/modules/gaussian_process.html
