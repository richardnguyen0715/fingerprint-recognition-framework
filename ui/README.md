# Fingerprint Recognition UI

A Streamlit-based interactive interface for the fingerprint recognition framework.

## Overview

This UI provides an intuitive way to interact with fingerprint recognition methods implemented in the core framework. It allows users to:

- **Verify** fingerprints (1:1 matching)
- **Analyze** matching scores and thresholds
- **Compare** multiple recognition methods on the same fingerprint pair

## Architecture

The UI follows a clean architecture where all fingerprint recognition logic resides in the core framework (`src/`). The UI acts purely as a presentation layer.

```
ui/
├── 🏠_Homepage.py                # Main entry point (Home page)
├── pages/                        # Streamlit multi-page files
│   ├── 1_🔐_Verification.py      # Verification page entry
│   ├── 2_📊_Analysis.py          # Analysis page entry
│   └── 3_⚖️_Comparison.py        # Comparison page entry
├── views/                        # Page view implementations
│   ├── verification.py           # 1:1 matching workflow
│   ├── analysis.py               # Score and threshold analysis
│   └── comparison.py             # Multi-method comparison
├── components/                   # Reusable UI components
│   ├── uploader.py               # Image upload widgets
│   ├── model_selector.py         # Model selection widgets
│   ├── config_panel.py           # Parameter configuration
│   ├── result_viewer.py          # Result display
│   └── comparison_table.py       # Comparison display
├── state/                        # Session state management
│   └── session_state.py          # Centralized state handling
├── utils/                        # Utility functions
│   ├── adapters.py               # Data format conversion
│   └── validation.py             # Input validation
└── README.md                     # This file
```

## Running the UI

```bash
# From the project root directory
streamlit run "ui/🏠_Homepage.py"
```

The application will open in your default web browser at `http://localhost:8501`.

## Features

### 1. Verification Page

The main workflow for comparing two fingerprints:

1. Upload two fingerprint images
2. Select a recognition method
3. Configure parameters (optional)
4. Run matching
5. View similarity score and detailed analysis

### 2. Analysis Page

Tools for understanding and analyzing matching scores:

- **Threshold Analysis**: Interactive threshold selection tool
- **Result History**: View history of matching results
- **Score Interpretation**: Guide to understanding scores

### 3. Method Comparison Page

Compare multiple recognition methods:

1. Upload fingerprint pair (or use images from verification)
2. Select multiple methods to compare
3. Run all methods
4. View comparative results in table, chart, and detailed formats
5. Export results as CSV

## Dynamic Model Discovery

The UI automatically discovers available matching methods from the registry. Adding a new method to the framework requires **zero UI code changes**.

### Adding a New Matcher

1. Implement your matcher in the core framework
2. Create an adapter that implements `BaseMatcher`
3. Register the adapter in `src/registry/adapters.py`

Example:

```python
# In src/registry/adapters.py

class MyMatcherAdapter(BaseMatcher):
    def __init__(self, param1: float = 1.0):
        self._param1 = param1
        self._create_matcher()
    
    @property
    def name(self) -> str:
        return "My Custom Matcher"
    
    @property
    def description(self) -> str:
        return "Description of what this matcher does."
    
    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> MatchResult:
        # Your matching logic
        score = self._matcher.compute_similarity(image_a, image_b)
        return MatchResult(
            score=score,
            details={"param1": self._param1},
        )
    
    def get_current_parameters(self) -> Dict[str, Any]:
        return {"param1": self._param1}

# Register the matcher
registry.register(
    matcher_id="my_matcher",
    name="My Custom Matcher",
    description="...",
    category="custom",
    factory=MyMatcherAdapter,
    parameters=[
        ParameterInfo(
            name="param1",
            display_name="Parameter 1",
            param_type=ParameterType.FLOAT,
            default=1.0,
            min_value=0.0,
            max_value=10.0,
        ),
    ],
)
```

The new matcher will automatically appear in the UI's model selector.

## Available Matchers

### Baseline Methods

| Method | Description |
|--------|-------------|
| **SSIM** | Structural Similarity Index - measures structural similarity |
| **MSE** | Mean Squared Error - pixel-wise comparison |
| **NCC** | Normalized Cross-Correlation - correlation coefficient |

### Minutiae-Based Methods

| Method | Description |
|--------|-------------|
| **Minutiae Matching** | RANSAC-based alignment and greedy matching |

### Descriptor Methods

| Method | Description |
|--------|-------------|
| **MCC** | Minutia Cylinder Code - local minutiae descriptors |

## Configuration

Matchers can be configured through the UI:

- **SSIM**: Window size, sigma, k1, k2 constants
- **Minutiae**: Distance threshold, angle threshold, RANSAC iterations
- **MCC**: Cylinder radius, spatial cells, angular sections, sigma values

## Dependencies

The UI requires:

- `streamlit>=1.28.0`
- `numpy`
- `pillow`
- `pandas`

Install with:

```bash
pip install streamlit numpy pillow pandas
```

## Design Principles

1. **Separation of Concerns**: UI contains no fingerprint recognition logic
2. **Dynamic Discovery**: Methods are discovered from registry, not hardcoded
3. **Component Reusability**: UI components are modular and reusable
4. **Session Persistence**: Images and results persist during session
5. **Type Safety**: Full type hints throughout the codebase

## Extending the UI

### Adding a New Page

1. Create a new module in `ui/pages/`
2. Implement a `render_*_page()` function
3. Add navigation entry in `ui/app.py`

### Adding a New Component

1. Create a new module in `ui/components/`
2. Implement component as a function returning values or `None`
3. Export from `ui/components/__init__.py`
