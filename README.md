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
data `sample.json` and model `model_rf.pickle`
```
uv run python request.py
```
