#!/bin/bash
export PYTHONPATH=src
uvicorn src.api:app --reload