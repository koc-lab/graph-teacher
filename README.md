<!-- markdownlint-disable MD013 -->

# GraphTeacher

The source for the _GraphTeacher: Transductive Fine-Tuning of Encoders through Graph Neural Networks_ paper, published in IEEE TAI.

GraphTeacher introduces a semi-supervised fine-tuning framework that augments Transformer encoders with Graph Neural Networks (GNNs). The method leverages unlabeled samples without accessing test nodes during graph construction (no leakage) and does not require re-graphing at inference, enabling scalable inductive deployment.

> This repository contains the official implementation accompanying the paper.

