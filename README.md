# Chess Engine with a Learned Evaluation

I built this project to compare a classic AI search that uses a hand-written heuristic with machine learning methods that learn from the Lichess evaluation database.

[Open in Colab](https://colab.research.google.com/github/ahmedgx1010/Chess-Engine/blob/main/Chess_engine.ipynb)

## How it works

**Search:** Minimax looks a few moves ahead, assuming both sides play their best move.

Alpha-beta pruning gives the same result but skips branches that can't change the decision.

**Evaluation:** how the engine scores a position. I compared four:

| Engine | Evaluation |
|---|---|
| Material | Counts pieces (pawn 1, knight/bishop 3, rook 5, queen 9) |
| Linear | Linear regression trained on Stockfish scores |
| MLP | Neural network (2 hidden layers: 128, 64) trained on Stockfish scores |
| Hybrid | Material count + MLP score |

## Data

- 500,000 positions from the [Lichess evaluation database](https://database.lichess.org/#evals), scored by Stockfish. 406,366 were kept (positions with a forced mate were skipped).
- Each board becomes 768 features: 12 piece types × 64 squares (1 = the piece is there).
- Scores are clipped to ±1000 centipawns so extreme values don't affect training too much. 80% of positions are used for training and 20% for testing.

## Results

### Prediction error (test set, lower is better)

| Model | Average error | Better than baseline |
|---|---|---|
| Baseline (always guess the average) | 171 cp | — |
| Linear regression | 158 cp | 8% |
| MLP, 100k positions | 136 cp | 20% |
| MLP, 325k positions | 117 cp | 32% |

### Piece values learned by the linear model (centipawns)

| Pawn | Knight | Bishop | Rook | Queen |
|---|---|---|---|---|
| 110 | 158 | 204 | 247 | 460 |

In my hand-written heuristic, I set the values to P = 1, N/B = 3, R = 5, Q = 9 and K = 0. The linear model learned the same order on its own; the only difference is that the bishop is slightly higher than the knight. The values are about half of the usual ones because I capped the scores at ±1000, and because many training positions are in the middle of a trade, where one side is temporarily a piece down but about to recapture.

### Engine matches (20 games each, depth 2, vs. Material)

| Engine | Wins | Draws | Losses | Score |
|---|---|---|---|---|
| MLP | 0 | 15 | 5 | 7.5 / 20 |
| Hybrid | 7 | 11 | 2 | 12.5 / 20 |

### Search speed (same position, depth 3)

| Minimax | Alpha-beta |
|---|---|
| 3.44 s | 0.26 s |

## What I learned

**1. Why the MLP was more accurate but still lost to simple material counting**

Even though the MLP was more accurate than simple material counting, it still lost, because its score focused on the quality of the position rather than on losing pieces. When I removed white's queen from the starting position, the material evaluation said −9, but the MLP said only −0.38. That's why I made the hybrid engine.

**2. Why the hybrid fixed it**

The hybrid combines the MLP's positional score with the material score, so it chooses good squares for its pieces without giving pieces away.

**3. What more data did to overfitting**

When I trained on only 100,000 positions, the loss went down and the validation score went up from the first iteration to the ninth. After that, the loss kept dropping until the twentieth iteration, but the validation score dropped too, which means the model was overfitting: it started memorizing the positions instead of learning from them. I solved this by using more data (about 325,000 positions), which raised the best validation score from 0.448 to 0.575.

## Limitations and next steps

- **Depth 2 only, because the ML evaluation is slow:**
  The search calls the evaluation thousands of times per move, and each call to the neural network takes much longer than counting pieces. On my setup, higher depths made each move take too long.
- **Most draws were the 150-move limit:**
  14 of the 15 draws in the MLP match and 10 of the 11 in the hybrid match ended at the move limit. Neither engine knows how to convert an advantage into checkmate, and that's something I want to research and solve.
- **The model doesn't know whose turn it is:**
  My function `board_to_features` only describes where the pieces are, so the models never receive this information. I plan to add it to the function in the future, which would likely lower the error.
- **Positions with a forced mate were left out of training:**
  For these positions, Stockfish doesn't give a centipawn score. It gives something like `mate: 3`, which means white can force checkmate in 3 moves (a negative number means black can). This isn't on the same scale as the other scores, so the simplest choice was to skip them. That was about 19% of the data, and it may be one reason the MLP undervalues big material advantages.

**Next steps:**
- Give the models the ability to know whose turn it is.
- Convert forced-mate positions to ±1000 and include them in training. This may also help the engines finish winning games instead of reaching the 150-move limit.

## How to run

Open the notebook in Colab and run all cells in order.

To play against an engine:
1. `show_engines()` to see the list of engines
2. `new_game(engine_number, depth=2)` to start a game
3. `my_move("e4")` to play your move (replace `e4` with your move)

To make two engines play each other:
`match(engine_a, engine_b, games=20, depth=2)`
