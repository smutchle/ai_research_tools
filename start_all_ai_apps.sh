#!/bin/bash

AI_BASE_DIR=`pwd`
CURRENT_DIR=`pwd`

cd $AI_BASE_DIR/references_bot/
./run_references_app_in_background.sh

cd $AI_BASE_DIR/research_rag_bot
./run_chatbot_in_background.sh

cd $AI_BASE_DIR/web_researcher_bot
./run_researcher_in_background.sh

cd $AI_BASE_DIR/knowledge_distillation
./run_kd_app_in_background.sh

cd $CURRENT_DIR
