import numpy as np

def extract_color_features(images):
    """
    Extract color-based features from eye conjunctiva images
    Features: Mean R, Mean G, Mean B, Redness Ratio
    """
    features = []

    for img in images:
        # Separate RGB channels
        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]

        # Compute mean values
        mean_R = np.mean(R)
        mean_G = np.mean(G)
        mean_B = np.mean(B)

        # Redness ratio (key anemia indicator)
        redness_ratio = mean_R / (mean_G + mean_B + 1e-6)

        # Store features
        features.append([mean_R, mean_G, mean_B, redness_ratio])

    return np.array(features)
