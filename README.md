# 🏥 NovaCare RAG Chatbot

## Overview

NovaCare Health Solutions Inc. is a fictional health insurance company
we created to build and evaluate a RAG (Retrieval-Augmented Generation)
chatbot. The company and all its policies, prices, and rules are entirely
made up — this guarantees the language model has never seen this data
during training, making it a clean testbed for measuring RAG's real impact.

The dataset contains 20 questions and answers covering NovaCare's plans,
coverage, claims, and policies. Our goal is to build a chatbot that can
accurately answer customer questions using only this document as its
knowledge base.

## Why a Fake Company?

A common mistake in RAG evaluation is using data the LLM already knows
from training. If the model can answer correctly without retrieval, you
cannot measure whether RAG is actually helping. By inventing NovaCare with
unique prices and specific policies, we guarantee the baseline LLM scores
zero without retrieval — making the value of RAG undeniable.

## The Data

The knowledge base is a single FAQ document containing 20 Q&A pairs covering
plans and premiums, deductibles and out-of-pocket maximums, coverage details
(mental health, ER, prescriptions, maternity), claims, appeals, cancellation
policies, wellness rewards, and telehealth services.

**Sample FAQ Entry**

Q1: What are NovaCare's available health insurance plans?
A: NovaCare offers three main plans: NovaCare Basic (monthly premium $187),
NovaCare Plus (monthly premium $334), and NovaCare Elite (monthly premium $521).
All plans include preventive care at no extra cost. NovaCare Elite additionally
covers dental and vision.

## Chunking Strategy

Most RAG tutorials chunk documents by character count — splitting every 500
characters with some overlap. This works for long prose but is wrong for FAQ data.

We use semantic chunking — each chunk is one complete Q&A pair. This produces
21 self-contained chunks (20 Q&A pairs + 1 header chunk). When the retriever
finds a chunk, it finds the full answer.

| Strategy | Semantic (ours) | Character-based |
|---|---|---|
| Chunk boundary | At each question | Every 500 chars |
| Each chunk | One complete Q&A | Random split |
| Context preserved | Always | Often split |
| Retrieval quality | High | Lower |