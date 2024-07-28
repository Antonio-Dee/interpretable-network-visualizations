# Interpretable Network Visualizations
All results obtained in the paper [*M. Bianchi, A. De Santis, A. Tocchetti and M. Brambilla: Interpretable Network Visualizations: A Human-in-the-Loop Approach for Post-hoc Explainability of CNN-based Image Classification*](https://www.ijcai.org/proceedings/2024/411) are provided in this repository

## Additional Results
In this folder we collected all images and visualizations for both global and local explanations for all classes used in the presented experiment.
- For local explanations, we provide an HTML file to ease the visualization process. With such, it is possible to select the class, the image and whether to apply the label merging algorithm or not. Additionally, all compressed explanations (i.e., heatmap, score and the top 3 labels) can be clicked to access the expanded visualization about the cluster map.
- For global explanations, we provide explanations by layer and by class in the form of png images.

## Code
In this folder we provide the code we used in the different phases of the experiment, including tests not presented in the paper (i.e., the code for different clustering algorithms and additional plots for label analysis).

## More about *Deep Reveal*
In this folder we collected:
- Screenshots of the *Deep Reveal* application
- An IFML diagram describing the pipeline of the application
- The form we utilized for evaluating the gamified activity

## Evaluation with SOTA
In this folder we present the form and the result we obtained by evaluating our method w.r.t. State-Of-The-Art Explainability techniques
