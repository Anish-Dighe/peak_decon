"""
Hybrid Peak Deconvolution Model

This module implements a hybrid approach:
1. Neural Network predicts initial parameters and number of components
2. Optimization refines the parameters for best fit

Model Architecture:
- CNN-based encoder for spectrum feature extraction
- Regression heads for parameter prediction
- Classification head for component number prediction
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize, differential_evolution
from peak_generator import GEGPeak


class PeakDeconvolutionCNN(nn.Module):
    """
    CNN model for initial parameter prediction

    Architecture:
    - 1D CNN to extract features from spectrum
    - Predicts number of components (1-10) as classification
    - Predicts parameters for up to 10 peaks (with masking)
    """

    def __init__(self, input_size=1000, max_peaks=10):
        super().__init__()
        self.max_peaks = max_peaks

        # CNN feature extractor
        self.conv1 = nn.Conv1d(1, 32, kernel_size=15, padding=7)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=15, padding=7)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=15, padding=7)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)

        # Calculate size after convolutions
        conv_output_size = input_size // 8  # 3 pooling layers

        # Fully connected layers
        self.fc1 = nn.Linear(128 * conv_output_size, 512)
        self.fc2 = nn.Linear(512, 256)

        # Classification head for number of components
        self.n_components_head = nn.Linear(256, max_peaks)  # Predict 1-10 components

        # Regression head for parameters (5 params per peak)
        # Output shape: (max_peaks, 5) for [alpha, tau, mu, sigma, amplitude]
        self.params_head = nn.Linear(256, max_peaks * 5)

    def forward(self, x):
        """
        Forward pass

        Parameters:
        -----------
        x : torch.Tensor, shape (batch_size, input_size)
            Input spectra

        Returns:
        --------
        tuple
            (n_components_logits, parameters)
            - n_components_logits: (batch_size, max_peaks) - logits for 1-10 components
            - parameters: (batch_size, max_peaks, 5) - parameters for each peak
        """
        # Add channel dimension
        x = x.unsqueeze(1)  # (batch, 1, input_size)

        # CNN feature extraction
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        # Flatten
        x = x.flatten(1)

        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        # Prediction heads
        n_components_logits = self.n_components_head(x)  # (batch, max_peaks)

        # Parameters with sigmoid/activation for proper ranges
        params_flat = self.params_head(x)  # (batch, max_peaks * 5)
        params = params_flat.view(-1, self.max_peaks, 5)  # (batch, max_peaks, 5)

        # Apply constraints to parameters
        params = self._apply_parameter_constraints(params)

        return n_components_logits, params

    def _apply_parameter_constraints(self, params):
        """
        Apply physical constraints to parameters

        Parameters format: [alpha, tau, mu, sigma, amplitude]
        Constraints:
        - alpha: (0.5, 3.0)
        - tau: (0.05, 0.3)
        - mu: (0.0, 1.0)
        - sigma: (0.01, 0.15)
        - amplitude: (0.0, 1.0)
        """
        constrained = torch.zeros_like(params)

        # alpha: 0.5 to 3.0
        constrained[:, :, 0] = torch.sigmoid(params[:, :, 0]) * 2.5 + 0.5

        # tau: 0.05 to 0.3
        constrained[:, :, 1] = torch.sigmoid(params[:, :, 1]) * 0.25 + 0.05

        # mu: 0.0 to 1.0 (peak position)
        constrained[:, :, 2] = torch.sigmoid(params[:, :, 2])

        # sigma: 0.01 to 0.15 (peak width)
        constrained[:, :, 3] = torch.sigmoid(params[:, :, 3]) * 0.14 + 0.01

        # amplitude: 0.0 to 1.0
        constrained[:, :, 4] = torch.sigmoid(params[:, :, 4])

        return constrained


class OptimizationRefiner:
    """
    Refine neural network predictions using optimization
    """

    def __init__(self, method='L-BFGS-B'):
        """
        Parameters:
        -----------
        method : str
            Optimization method: 'L-BFGS-B', 'differential_evolution', etc.
        """
        self.method = method

    def refine(self, spectrum_x, spectrum_y, initial_params, n_components):
        """
        Refine parameters using optimization

        Parameters:
        -----------
        spectrum_x : array
            X values of spectrum
        spectrum_y : array
            Y values of spectrum to fit
        initial_params : array, shape (n_components, 5)
            Initial parameter guesses from neural network
        n_components : int
            Number of components

        Returns:
        --------
        array, shape (n_components, 5)
            Refined parameters
        """
        # Flatten parameters for optimization
        x0 = initial_params.flatten()

        # Define bounds for each parameter type
        bounds = []
        for _ in range(n_components):
            bounds.extend([
                (0.5, 3.0),    # alpha
                (0.05, 0.3),   # tau
                (0.0, 1.0),    # mu
                (0.01, 0.15),  # sigma
                (0.0, 2.0)     # amplitude (allow higher for optimization)
            ])

        # Objective function: minimize MSE
        def objective(params_flat):
            params_2d = params_flat.reshape(n_components, 5)
            predicted_y = self._generate_spectrum(spectrum_x, params_2d)
            mse = np.mean((spectrum_y - predicted_y) ** 2)
            return mse

        # Optimize
        if self.method == 'differential_evolution':
            result = differential_evolution(objective, bounds, maxiter=100, seed=42)
        else:
            result = minimize(objective, x0, method=self.method, bounds=bounds,
                            options={'maxiter': 500})

        # Reshape result
        refined_params = result.x.reshape(n_components, 5)

        return refined_params, result.fun

    def _generate_spectrum(self, x, params):
        """Generate spectrum from parameters"""
        y = np.zeros_like(x)
        for param in params:
            peak = GEGPeak(
                alpha=param[0],
                tau=param[1],
                mu=param[2],
                sigma=param[3],
                amplitude=param[4]
            )
            _, peak_y = peak.generate_peak(x)
            y += peak_y
        return y


class HybridDeconvolutionModel:
    """
    Complete hybrid model combining CNN and optimization
    """

    def __init__(self, model_path=None, device='cpu'):
        """
        Parameters:
        -----------
        model_path : str, optional
            Path to saved model weights
        device : str
            'cpu' or 'cuda'
        """
        self.device = device
        self.cnn_model = PeakDeconvolutionCNN(input_size=1000, max_peaks=10)
        self.cnn_model.to(device)

        if model_path is not None:
            self.load_model(model_path)

        self.optimizer_refiner = OptimizationRefiner(method='L-BFGS-B')

    def predict(self, spectrum_y, spectrum_x=None, use_optimization=True,
                confidence_threshold=0.5):
        """
        Predict peak parameters for a spectrum

        Parameters:
        -----------
        spectrum_y : array
            Spectrum intensity values
        spectrum_x : array, optional
            Spectrum position values (assumed normalized 0-1)
        use_optimization : bool
            Whether to refine predictions with optimization
        confidence_threshold : float
            Minimum confidence for component prediction

        Returns:
        --------
        tuple
            (n_components, parameters, confidence, metrics)
        """
        if spectrum_x is None:
            spectrum_x = np.linspace(0, 1, len(spectrum_y))

        # Normalize input
        spectrum_y = np.asarray(spectrum_y)
        if spectrum_y.max() > 0:
            spectrum_y = spectrum_y / spectrum_y.max()

        # Neural network prediction
        self.cnn_model.eval()
        with torch.no_grad():
            spectrum_tensor = torch.FloatTensor(spectrum_y).unsqueeze(0).to(self.device)
            n_comp_logits, params = self.cnn_model(spectrum_tensor)

            # Get predicted number of components
            n_comp_probs = F.softmax(n_comp_logits, dim=1)[0].cpu().numpy()
            n_components = int(np.argmax(n_comp_probs)) + 1  # +1 because 0-indexed
            confidence = float(n_comp_probs[n_components - 1])

            # Get parameters for predicted components
            initial_params = params[0, :n_components, :].cpu().numpy()

        # Sort by peak position
        sort_idx = np.argsort(initial_params[:, 2])
        initial_params = initial_params[sort_idx]

        # Optimization refinement
        if use_optimization:
            refined_params, final_loss = self.optimizer_refiner.refine(
                spectrum_x, spectrum_y, initial_params, n_components
            )
            final_params = refined_params
        else:
            final_params = initial_params
            final_loss = None

        # Calculate goodness of fit metrics
        metrics = self._calculate_metrics(spectrum_x, spectrum_y, final_params)

        return n_components, final_params, confidence, metrics

    def _calculate_metrics(self, x, y_true, params):
        """Calculate goodness of fit metrics"""
        y_pred = np.zeros_like(y_true)
        for param in params:
            peak = GEGPeak(
                alpha=param[0],
                tau=param[1],
                mu=param[2],
                sigma=param[3],
                amplitude=param[4]
            )
            _, peak_y = peak.generate_peak(x)
            y_pred += peak_y

        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # RMSE
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

        # Normalized RMSE
        nrmse = rmse / (y_true.max() - y_true.min()) if y_true.max() > y_true.min() else 0

        return {
            'R2': r2,
            'RMSE': rmse,
            'NRMSE': nrmse,
            'residual_sum': ss_res
        }

    def save_model(self, path):
        """Save model weights"""
        torch.save(self.cnn_model.state_dict(), path)
        print(f"Model saved to: {path}")

    def load_model(self, path):
        """Load model weights"""
        self.cnn_model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from: {path}")


if __name__ == '__main__':
    print("Hybrid Deconvolution Model Demo")
    print("=" * 60)

    # Create model
    print("\n1. Creating hybrid model...")
    model = HybridDeconvolutionModel(device='cpu')
    print(f"Model created with {sum(p.numel() for p in model.cnn_model.parameters())} parameters")

    # Generate test spectrum
    print("\n2. Generating test spectrum...")
    from peak_generator import generate_random_spectrum
    x, y, true_params = generate_random_spectrum(n_components=3, seed=42)
    print(f"True parameters:\n{true_params}")

    # Test prediction (without trained model, results will be random)
    print("\n3. Testing prediction pipeline...")
    print("NOTE: Model is untrained, predictions will be random!")
    n_comp, pred_params, confidence, metrics = model.predict(y, x, use_optimization=False)
    print(f"Predicted components: {n_comp} (confidence: {confidence:.3f})")
    print(f"Predicted parameters:\n{pred_params}")
    print(f"Metrics: R2={metrics['R2']:.3f}, RMSE={metrics['RMSE']:.4f}")

    print("\n" + "=" * 60)
    print("Model architecture is ready!")
    print("Next: Train model with synthetic data using train.py")
