
FERE-CRS: A Calculus of Semantic Inference for Robust, Adaptive Reasoning in AI Agents
This repository contains the final manuscript, experimental data, and source code for the FERE-CRS (Free Energy Resonance - Cognitive Resonance Score) research program. This multi-year project was dedicated to developing a new class of cognitive architecture for Large Language Model (LLM) based agents, grounded in the principles of Active Inference.
Our work has culminated in the v9.0 Integrated Agent, a prototype that demonstrates robust, generalizable, and adaptive reasoning. This agent is guided by a Calculus of Semantic Inference, a principled framework for meta-cognitive control that allows the agent to intelligently manage its conceptual boundaries by knowing when to be certain and when to be curious.
This repository provides all the necessary materials to understand and reproduce the final, successful validation experiment detailed in our paper.
Repository Structure
The repository is organized to provide a clear and direct path to understanding our research.
•	MANUSCRIPT.pdf: The final, comprehensive paper detailing the theoretical foundations, multi-phase research journey, and final results of the FERE-CRS I & II programs.
•	requirements.txt: A file listing all Python dependencies required to run the final experiment.
•	/CODE/: This directory contains the definitive source code for our final agent.
o	phase_xiv_harness_v9.0.py: The final, successful v9.0 Integrated Agent and the test harness script used to run the validation suite.
•	/DATA/: This directory contains the input data for the final experiment.
o	/test_suites/phase_xiv_test_suite.json: The Generalization & Robustness Test Suite, containing all the problem scenarios.
•	/RESULTS/: This directory contains the raw output from our definitive experimental run.
o	phase_xiv_v9_final_results.log: The complete, unedited log file from the final validation run, providing the ground-truth evidence for the results reported in the manuscript.
•	/ARCHIVE/: This directory provides deep historical context for the project.
o	FERE-CRS_Living_Document_v14.0.md: The complete, phase-by-phase technical summary of the entire FERE-CRS I & II research programs, detailing the iterative, failure-driven methodology that led to the final architecture.
Reproducing the Experiment
To replicate the final validation experiment (Phase XIV), please follow these steps.
1. Setup the Environment
First, clone this repository to your local machine.
git clone https://github.com/ThomasDevitt/FERE-CRS.git
cd FERE-CRS_Project

Next, install the required Python dependencies.
pip install -r requirements.txt

Finally, create a .env file in the root of the project directory to securely store your Google AI API key. The file should contain a single line:
GOOGLE_API_KEY="YOUR_API_KEY_HERE"

2. Run the Experiment
The phase_xiv_harness_v9.0.py script is a self-contained test harness that will automatically load the test suite from the /DATA/ directory and run all experimental cases.
Execute the script from the root of the project directory:
python CODE/phase_xiv_harness_v9.0.py

The script will print its progress to the console, detailing the agent's P-CRS scores and judgments for each piece of evidence. The final output in your console should match the results documented in the phase_xiv_v9_final_results.log file located in the /RESULTS/ directory.

Citing Our Work
If you use this research or the FERE-CRS architecture in your work, please cite the manuscript. The complete research protocol, including this manuscript, has been permanently archived on Zenodo.

Devitt, T. E., & Gemini. (2025). FERE-CRS: A Calculus of Semantic Inference for Robust, Adaptive Reasoning in AI Agents. Zenodo. 10.5281/zenodo.16923559

Contact
For questions, collaborations, or discussions, please contact Thomas E. Devitt.

