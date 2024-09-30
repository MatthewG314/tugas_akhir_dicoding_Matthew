## Setup Environment

### Using Anaconda
```bash
conda create --name bike-sharing-ds python=3.9
conda activate bike-sharing-ds
pip install -r requirements.txt


### Using Shell/Terminal
mkdir bike_sharing_analysis
cd bike_sharing_analysis
pipenv install
pipenv shell
pip install -r requirements.txt

### Run Steamlit app
streamlit run dashboard.py

