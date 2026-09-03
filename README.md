\# Network Attack Forecasting



An AI-based temporal forecasting system for predicting whether a network attack is likely to occur within the next 15 minutes using recent network-flow behavior.



\## Overview



This project focuses on \*\*network attack forecasting\*\*, rather than only classifying attacks that have already occurred.



The system observes a rolling history of network activity and predicts whether an attack will occur during a future 15-minute horizon.



\### Forecasting setup



\- Historical context: \*\*10 minutes\*\*

\- Forecast horizon: \*\*15 minutes\*\*

\- Prediction target: attack occurrence during the next 15 minutes

\- Primary model: \*\*XGBoost\*\*

\- Neural-network baseline: \*\*GRU\*\*

\- Enhanced feature count: \*\*161\*\*

\- Flattened model input: \*\*1,610 features\*\*



The overall pipeline is:



```text

Network Flow Data

&#x20;      |

&#x20;      v

Temporal Aggregation

&#x20;      |

&#x20;      v

Enhanced Feature Engineering

&#x20;      |

&#x20;      v

10-Minute Historical Window

&#x20;      |

&#x20;      v

XGBoost Forecasting Model

&#x20;      |

&#x20;      v

Attack Probability

&#x20;      |

&#x20;      v

Risk Level

&#x20;      |

&#x20;      v

15-Minute Attack Forecast

