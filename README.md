# hw7
- Homework 6
- November 11, 2025
- Cherryl Chico, Xianrui Cao, Nikoloz Darsalia, Xiaoyan Wang


1. To run, sync environment
`uv sync`

2. Activate environment
`source .venv/bin/activate`

3. Run local server
`uv run uvicorn main:app --reload`

4. In another terminal, run the request.py to make a prediction for
data `sample.json` and model `model_rf.pickle`
`uv run python request.py`
