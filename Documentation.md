# ML Challenge 2025: Smart Product Pricing Solution Template 
 
 Team Name: B1TFORCE  
 Team Members: JISHNU NANDAKUMAR, ADIT U.A, SWAYAM VIJAY MEHRA  
 Submission Date: 13-10-2025
 
 --- 
 
 ## 1. Executive Summary 
 We implement a multimodal pricing model that combines text embeddings from product catalog content with visual features from product images. These features feed a `GradientBoostingRegressor`, producing positive price predictions formatted per the challenge requirements. The pipeline is efficient, reproducible, and leverages robust pre-trained models for feature extraction. 
 
 --- 
 
 ## 2. Methodology Overview 
 
 ### 2.1 Problem Analysis 
 We treat pricing as a supervised regression problem influenced by product attributes present in text and images. Text captures brand, specifications, and Item Pack Quantity (IPQ) cues; images add perceived quality and design indicators. We ensure the output format exactly matches `sample_test_out.csv`. 
 
 *Key Observations:* 
 - Text descriptions often encode brand, specs, and IPQ, which correlate with price. 
 - Images provide complementary signals of quality and presentation. 
 - A unified feature vector of text and image embeddings improves predictive power. 
 
 ### 2.2 Solution Strategy 
 *Approach Type:* Single Model (GradientBoostingRegressor) with multimodal features  
 *Core Innovation:* Effective integration of `SentenceTransformer` text embeddings and `ResNet18` image features, trained with a gradient boosting regressor on unified embeddings. 
 
 --- 
 
 ## 3. Model Architecture 
 
 ### 3.1 Architecture Overview 
 1)Load and inspect the training and test datasets.
 2)Preprocess catalog content to handle missing values and extract text embeddings using all-MiniLM-L6-v2.
 3)Download and preprocess images using ImageNet standards, and extract features with pretrained ResNet18.
 4)Concatenate text and image embeddings into a unified feature representation.
 5)Train a GradientBoostingRegressor on the combined embeddings.
 6)Predict test prices and generate test_out.csv with formatted positive price values.
 
 ### 3.2 Model Components 
 
 *Text Processing Pipeline:* 
 - Preprocessing steps: [Fill missing `catalog_content` with empty string, strip whitespace] 
 - Model type: [SentenceTransformer `all-MiniLM-L6-v2`] 
 - Key parameters: [embedding_dim=384, `show_progress_bar`=True] 
 
 *Image Processing Pipeline:* 
 - Preprocessing steps: [Resize(256), CenterCrop(224), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])] 
 - Model type: [`torchvision.models.resnet18(pretrained=True)`, eval mode] 
 - Key parameters: [input_size=224, output_dim≈1000 (logits), image download via `src/utils.download_images`] 
 
 *Final Regression Model:* 
 - Model type: [`sklearn.ensemble.GradientBoostingRegressor`] 
 - Key parameters: [n_estimators=200, max_depth=6, validation split=10%, random_state=42] 
 - Feature selection: [All columns starting with `text_emb_` or `image_emb_`] 
 
 --- 
 
 ## 4. Model Performance 
 
 ### 4.1 Validation Results 
- **Mean Absolute Error (MAE):** 17.3154
- **Symmetric Mean Absolute Percentage Error (SMAPE):** 13.21%
- **Root Mean Squared Error (RMSE):** 22.7422
- **R² (Coefficient of Determination):** 0.5723

 
 ## 5. Conclusion 
 The approach integrates strong pre-trained text and image encoders with a gradient boosting regressor, yielding reliable price predictions while enforcing positive outputs and correct formatting. The design is straightforward, reproducible, and aligned with the challenge’s constraints. 
 
 --- 
 
 ## Appendix 
 
 ### A. Code artefacts 
    Complete source code, trained models, and datasets are available at:
    🔗 https://drive.google.com/drive/folders/1cFycAWCBrgt7dpJjRHgNcwUiuIl7yYba?usp=sharing

    Included Files:
    Core scripts: data_preprocessing.py, model.py, predict.py, src/utils.py
    Trained model: product_price_model.pkl
    Dataset: dataset/train.csv, dataset/test.csv, dataset/*_features.csv
    Sample I/O: dataset/sample_test.csv, dataset/sample_test_out.csv

 ### B. Additional Results  
 ### Predicted vs Actual Prices

The "Predicted vs Actual Prices" plot is available at:
[View Chart](https://drive.google.com/file/d/1H-UI8aoRLj11jy8sNMXy6LN3imx4vw70/view?usp=sharing)

 
 