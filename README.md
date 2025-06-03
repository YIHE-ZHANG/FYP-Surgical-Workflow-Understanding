# Bridge-Prompt for Surgical Action Recognition

This project represents a constraint-aware adaptation of the [Bridge-Prompt](https://github.com/ttlmh/Bridge-Prompt) framework, specifically engineered for fine-grained surgical action recognition in robot-assisted radical prostatectomy (RARP) procedures. The implementation builds upon the foundational work presented in the original Bridge-Prompt research.

---

## System Compatibility and Platform Considerations

An important consideration for researchers planning to work with this implementation concerns platform compatibility. Due to development environment constraints and system configuration requirements, this repository provides Windows-based command implementations exclusively. While equivalent Linux commands exist for most operations, they have not undergone comprehensive testing within this specific deployment context.

Linux users will need to adapt the provided Windows batch scripts to their shell environments. This typically involves:

- Converting `.bat` files to `.sh` equivalents  
- Adjusting path separator conventions from backslashes (`\`) to forward slashes (`/`)  
- Modifying environment activation commands to match their package management approach  

The underlying Python code remains platform-agnostic, so the adaptation primarily concerns the convenience scripts that orchestrate training and evaluation workflows.

---

## Understanding the Computational Context

This implementation was deliberately designed to operate within realistic resource constraints that mirror those encountered by individual researchers and smaller institutions. The target hardware specification centers around NVIDIA RTX 3060 GPUs with 16 GB of memory, representing a deliberate departure from high-resource implementations that assume access to enterprise-grade computational infrastructure.

This constraint-aware approach influences every aspect of the system design, from memory management strategies to batch-size optimization and training checkpoint frequency. The goal is to demonstrate that sophisticated vision-language models can achieve meaningful performance within the computational boundaries that characterize real-world academic and clinical environments.

---

## Prerequisites and Environment Configuration

Before beginning work with this implementation, establishing a properly configured computational environment represents a critical first step. The system depends on several interconnected frameworks that must be carefully coordinated to ensure stable operation throughout training and evaluation cycles.

### Essential Software Dependencies

The foundation of this implementation rests on PyTorch with CUDA acceleration, which provides the computational substrate for both the vision and language components of the Bridge-Prompt architecture. Installing these dependencies requires attention to version compatibility, particularly between CUDA toolkit versions and PyTorch builds:

```bash
# Install PyTorch with CUDA 11.8 support for optimal GPU acceleration
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install CLIP implementation for vision-language model backbone
pip install clip-by-openai

# Install experiment tracking and logging infrastructure
pip install wandb

# Install configuration management and data handling utilities
pip install dotmap
pip install pyyaml
pip install tqdm

# Install scientific computing and visualization libraries
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install pandas
pip install numpy
```

> **Note:** The dependency installation process benefits significantly from using a dedicated `conda` environment, which isolates these specific requirements from other projects and system-wide packages. This isolation proves particularly valuable when working with CUDA-enabled PyTorch installations, where version conflicts can lead to subtle performance degradation or complete training failures.

Creating an isolated environment also facilitates reproducibility across different development machines and enables easier sharing of the computational setup with other researchers who may want to build upon this work.

---

## Dataset Organization and Structure

The RARP-50 dataset requires careful organization to ensure compatibility with the data loading infrastructure built into this implementation. Understanding the expected directory structure helps prevent common configuration issues that can derail training before it begins.

### Required Directory Hierarchy

The data loading system expects a hierarchical organization that separates training and testing video collections while maintaining a consistent internal structure for individual video sequences. Each video directory contains both the visual content (RGB frames) and the corresponding temporal action annotations that provide supervision for the learning process:

```
data/
├── rarp50/
│   ├── train/
│   │   ├── video_01/
│   │   │   ├── rgb/
│   │   │   │   ├── 000001.jpg
│   │   │   │   ├── 000002.jpg
│   │   │   │   └── ... (additional frame files)
│   │   │   └── action_discrete.txt
│   │   ├── video_02/
│   │   │   └── ... (same structure as video_01)
│   │   └── ... (additional training videos)
│   └── test/
│       ├── video_xx/
│       │   └── ... (same structure as training videos)
│       └── ... (additional test videos)
```

- Each `video_xx/rgb/` folder contains sequential JPEG frames (`000001.jpg`, `000002.jpg`, etc.).  
- Each `action_discrete.txt` file is a CSV with lines formatted as `timestamp, class_id`. This provides frame-level supervision while preserving the temporal continuity essential for understanding surgical workflow dynamics.

Understanding this organization helps researchers verify that their dataset preparation process has completed successfully and can help diagnose data loading issues that might arise during training initialization.

---

## Training Process and Optimization Strategy

The training pipeline incorporates several sophisticated strategies designed to maximize performance within the resource constraints that define this implementation. These strategies work together to address the multiple challenges presented by surgical action recognition, including extreme class imbalance, fine-grained visual distinctions, and temporal sequence modeling.

### Initiating the Training Process

The primary training command launches a comprehensive optimization process that handles model initialization, data loading, prompt engineering, and iterative parameter updates while maintaining careful attention to memory management throughout the training cycle:

```bat
batchscripts\run_train.bat .\configs\rarp50\rarp50_ft.yaml
```

- Activates the complete training infrastructure, including class-weighted loss functions designed to address the 18.5 : 1 imbalance ratio present in surgical workflow data.  
- Incorporates domain-specific data augmentations that simulate realistic surgical visual conditions and enhance model robustness to authentic intraoperative variability.  
- Uses configuration parameters calibrated for NVIDIA RTX 3060 memory boundaries, balancing training effectiveness against computational feasibility and preventing out-of-memory errors.

### Checkpoint-Based Training Continuation

Long training cycles within resource-constrained environments benefit significantly from robust checkpoint management that enables recovery from interruptions and systematic exploration of different training durations. The implementation provides comprehensive checkpoint functionality that preserves model weights, optimizer states, and learning rate schedules:

```bat
batchscripts\run_train.bat .\configs\rarp50\rarp50_ft.yaml resume "path\to\checkpoint.pt"
```

- Resuming from a checkpoint maintains optimization momentum, ensuring that resumed training continues smoothly from the exact state where the previous session terminated.  
- Checkpoints also allow researchers to compare different training durations and hyperparameter settings without restarting from scratch.

### Exploring Available Checkpoints

Before attempting to resume training from a previous session, researchers can systematically explore the available checkpoint files to identify the most appropriate restoration point:

```bat
batchscripts\run_train.bat .\configs\rarp50\rarp50_ft.yaml list
```

- Lists all saved checkpoints along with creation timestamps and metadata.  
- Enables informed decision-making about which checkpoint represents the optimal starting point for continued experimentation.

---

## Model Evaluation and Performance Assessment

The evaluation infrastructure provides comprehensive assessment capabilities that extend far beyond simple accuracy metrics to include detailed analysis of model behavior across different surgical action categories. This multi-dimensional evaluation approach is essential for understanding model performance in the context of clinical applicability.

### Comprehensive Model Testing

Once training has progressed to a satisfactory checkpoint, the evaluation process can be initiated to generate detailed performance metrics and prediction outputs:

```bat
batchscripts\run_test.bat .\configs\rarp50\rarp50_test.yaml
```

- Generates predictions for the complete test set while maintaining detailed records of model confidence, per-class performance characteristics, and temporal prediction patterns.  
- Uses evaluation parameters optimized for memory-constrained hardware, ensuring that evaluation can be performed on the same machine that supported training.  
- Produces output files that summarize performance, which can be used for both quantitative and qualitative analysis.

### Detailed Results Analysis

Following the completion of model evaluation, specialized analysis tools provide deeper insights into model performance characteristics, helping researchers understand both strengths and limitations:

```bash
batchpython analyze_rarp50_results.py
```

- Generates confusion matrices, per-class precision and recall metrics, and statistical summaries.  
- Examines temporal prediction patterns to identify how the model handles sequence dynamics.  
- Focuses on minority class performance, recognizing that rare but clinically critical actions often represent the most challenging aspects of surgical workflow understanding.

---

## Dataset Analysis and Class Distribution Understanding

The extreme class imbalance present in authentic surgical workflow data represents one of the most significant challenges for machine learning approaches in this domain. Understanding these distributional characteristics provides essential context for interpreting model performance and designing appropriate mitigation strategies.

### Comprehensive Statistical Analysis

The dataset analysis utility provides detailed insights into the class distribution patterns that fundamentally shape the learning problem:

```bash
batchpython analyze_rarp50_classes.py
```

- Outputs quantitative metrics (e.g., imbalance ratios, class frequencies) that quantify the severity of class imbalance.  
- Produces visualizations of class frequency distributions to highlight underrepresented actions.  
- Offers guidance on addressing imbalance through class weighting or alternative sampling strategies, while preserving clinical authenticity.

---

## Configuration Management and System Customization

The modular configuration approach employed by this implementation separates different aspects of system behavior into manageable, modifiable components that can be adjusted without requiring code changes. This design philosophy enhances both accessibility and experimental flexibility.

### Understanding Configuration Parameters

Configuration files are located in the `configs/rarp50/` directory:

- `rarp50_ft.yaml` – Training-specific parameters (e.g., batch size, learning rate, data augmentation settings, class-weighted loss coefficients).  
- `rarp50_test.yaml` – Evaluation-focused settings (e.g., batch size for testing, confidence thresholds, output directories).

These configurations reflect a careful balance between computational efficiency and model effectiveness. Key parameters include:

- **`batch_size`** (impacts memory usage)  
- **`num_workers`** (affects data loading efficiency)  
- **`frame_sampling_rate`** (controls temporal resolution)  

Researchers can modify these values via the YAML interface without changing underlying code.

### Hardware-Specific Adaptations

Researchers working with different GPU configurations may need to adjust parameters to match available computational resources. The modular structure supports adaptation to both more powerful and more limited hardware environments. Parameters likely to require adjustment include:

- **Batch sizes** (reduce to prevent out-of-memory errors)  
- **Number of data loader workers** (optimize disk I/O vs. CPU usage)  
- **Frame sampling frequency** (trade off temporal detail vs. computational cost)

---

## Troubleshooting Common Implementation Challenges

Working with resource-constrained deep learning implementations often presents unique challenges that differ from those encountered in high-resource research environments. Understanding these potential issues and their solutions helps researchers navigate the implementation successfully.

### Memory Management Challenges

- **Issue:** Training terminates with CUDA out-of-memory errors despite optimized configuration.  
  **Solution:** Reduce the `batch_size` parameter in the configuration file. You may also need to adjust learning rate schedules to maintain optimization effectiveness.  
  The implementation includes periodic memory cleanup routines and checkpoint optimization, but extreme memory pressure may still require manual intervention. Monitor GPU memory usage (e.g., via `nvidia-smi`) during initial runs to identify sustainable operating parameters.

### Dependency Configuration Issues

- **Issue:** Subtle performance degradation or unexpected behavior during long training runs due to version mismatches between CUDA toolkit, PyTorch, and drivers.  
  **Solution:** Use a dedicated `conda` environment to isolate dependencies. If issues arise, recreate the environment from scratch rather than attempting to resolve individual package conflicts.

### Data Loading and Path Resolution

- **Issue:** Data loading fails due to incorrect directory structure or misformatted annotation files.  
  **Solution:** Manually verify that the dataset organization matches the expected hierarchy (`data/rarp50/train/video_xx/rgb` and `action_discrete.txt`). Check file naming conventions, directory consistency, and CSV formatting in `action_discrete.txt`. The data loader’s error reporting can help identify missing files or misnamed paths.

---

## Project Architecture and Implementation Philosophy

Understanding the overall project organization helps researchers navigate the codebase effectively and identify appropriate entry points for modifications or extensions. The implementation follows a modular design philosophy that separates different concerns into distinct, manageable components.

- **`train.py`**: Orchestrates the full training pipeline, from data loading and prompt engineering to model optimization and checkpoint management. Integrates class weighting, memory management, and data augmentation strategies.  
- **`test.py`**: Handles evaluation and prediction generation. Produces metrics and output files used for detailed performance analysis.  
- **`batchscripts/`**: Contains Windows `.bat` scripts for launching training, resuming from checkpoints, listing checkpoints, and running evaluations.  
- **`configs/rarp50/`**: Houses YAML configuration files (`rarp50_ft.yaml`, `rarp50_test.yaml`) that encode hyperparameters and system settings.  
- **`analyze_rarp50_results.py`**: Provides utilities for analyzing evaluation outputs, generating confusion matrices, per-class precision/recall, and other statistical summaries.  
- **`analyze_rarp50_classes.py`**: Offers tools for inspecting class distribution within the RARP-50 dataset, quantifying imbalance, and producing visualizations.

Supporting utilities for dataset analysis, results interpretation, and configuration management enhance the research workflow without complicating primary training and evaluation pathways. This modular approach facilitates both immediate use and future extension of the implementation.

---

## Future Directions and Extensibility

This constraint-aware implementation provides a solid foundation for exploring alternative approaches to surgical action recognition within resource-limited environments. The modular design enables systematic experimentation with different prompt engineering strategies, class weighting schemes, and architectural modifications without requiring comprehensive system redesign.

Potential future directions include:

- Extending the prompt engineering framework to incorporate more sophisticated medical terminology.  
- Experimenting with alternative class balancing approaches (e.g., focal loss, oversampling rare classes).  
- Adapting the architecture for different surgical procedures or medical domains (e.g., cholecystectomy, hysterectomy).  
- Integrating multimodal data sources (e.g., kinematic signals from robotic tools, audio commentary).  
- Evaluating transfer learning from larger surgical datasets to improve rare action recognition.

The emphasis on accessibility and reproducibility built into this implementation ensures that future extensions can maintain the same philosophy of constraint-aware development while building upon the established foundation of working, documented, and tested components.

---

## License

This project is released under the MIT License.  
See [LICENSE](LICENSE) for details.
