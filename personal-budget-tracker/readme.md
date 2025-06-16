# Personal Budget Tracker - Simple Guide

## What You Need to Build
A program that tracks monthly income and expenses, then shows how much money is left.

## Basic Structure

### Step 1: Collect Information
- Ask user for their monthly income (one number)
- Ask for expense categories and amounts:
  - Rent: $800
  - Food: $300
  - Entertainment: $150
- Store everything in a dictionary

### Step 2: Do the Math
- Add up all expenses
- Subtract total expenses from income
- Calculate what percentage each category takes from income

### Step 3: Show Results
Display something like:
Income: $2000
Total Expenses: $1250
Money Left: $750
Food is 15% of your income

### Step 4: Save Data
- Write the budget info to a text file
- Later, read it back to load previous budgets

## Build It Step by Step

1. **Start Simple**: Just income and 3 expense types
2. **Add Calculations**: Total expenses and remaining money
3. **Make It Pretty**: Format the display nicely
4. **Add File Saving**: Store data in a file
5. **Add Menu**: Let users choose what to do

## Key Programming Concepts Used
- **Variables**: Store income and expense amounts
- **Dictionaries**: Organize expense categories
- **Functions**: Separate tasks (input, calculate, display)
- **Loops**: Ask for multiple expense categories
- **File Operations**: Save and load budget data

The goal is to create something useful while learning basic Python concepts through a real-world problem.
