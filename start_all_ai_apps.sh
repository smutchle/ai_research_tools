#!/bin/bash

AI_BASE_DIR=`pwd`
CURRENT_DIR=`pwd`

eval "$(conda shell.bash hook)"
conda activate genai

cd $AI_BASE_DIR/research_rag_bot
./run_chatbot_in_background.sh

cd $CURRENT_DIR
