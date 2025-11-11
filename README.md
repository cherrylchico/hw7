# Homework 7
- November 11, 2025
- Cherryl Chico, Xianrui Cao, Nikoloz Darsalia, Xiaoyan Wang

## How to Run

1. Sync environment
```
uv sync
```

2. Activate environment
```
source .venv/bin/activate
```

3. Run local server
```
uv run uvicorn main:app --reload
```

4. In another terminal, run request.py to make a prediction for
data `sample.json` using `model_rf.pickle`
```
uv run python request.py
```
The request should return the predicted probability.

5. Stop the server by pressing `CTRL` + `C` in the server terminal.

P.S for illustrating Pydantic data validation, when the input is not correct, run `wrong_request.py`, which uses `wrong_sample.json`.