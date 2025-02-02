# `PoFINN`: Integrating Physics of Failure and Neural Networks for Long-term Degradation Forecasting of Aluminum Electrolytic Capacitors


The work introduces `PoFINN`, a novel Physics-Informed Neural Network (`PINN`) framework for predicting the _long-term degradation_ of _Aluminum Electrolytic Capacitors_ (AECs) under storage conditions. By integrating domain knowledge of electrolyte evaporation, `PoFINN` combines _physics-based_ and _data-driven_ approaches, offering superior accuracy and interpretability. The framework operates in two phases—**Pre-Training** and **Real-Time Training**—using a hybrid loss function that incorporates both physical and data-driven constraints. `PoFINN` outperforms traditional methods in long-term predictions, especially for complex, non-linear degradation behaviors, enhancing predictive maintenance and reliability.


![An aluminium electrolytic capacitor](FIGS/Capacitor_diag_.jpg)

Following is the architectural view of the `PoFINN` model. 

![Architectural view of the Physics of Failure inspired Neural Network.](FIGS/architecture.jpg)


Following are the true versus predicted forecasts for the capacitors C~1~, C~2~, C~3~, and C~4~.

![True versus Predicted forecasts for the capacitors C~1~, C~2~, C~3~, and C~4~.](RES/c1-4.jpg)
![True versus Predicted forecasts for the capacitors C~5~, C~6~, C~7~, and C~8~.](RES/c5-8.jpg)#   P o F I N N  
 