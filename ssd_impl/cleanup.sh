#!/bin/bash
sudo sysctl -w vm.drop_caches=1
sudo swapoff -a
sudo swapon swap.img
