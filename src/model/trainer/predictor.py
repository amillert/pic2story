import torch


class Predictor:
    def __init__(self):
        pass

    def predict(self):
        MODEL.eval()

        state_h, state_c = net.zero_state(1)
        state_h = state_h.to(device)
        state_c = state_c.to(device)
        for w in words:
            ix = torch.tensor([[vocab_to_int[w]]]).to(device)
            output, (state_h, state_c) = net(ix, (state_h, state_c))

        _, top_ix = torch.topk(output[0], k=top_k)
        choices = top_ix.tolist()
        choice = np.random.choice(choices[0])

        words.append(int_to_vocab[choice])

    def eval(self):
        for _ in range(100):
            ix = torch.tensor([[choice]]).to(device)
            output, (state_h, state_c) = net(ix, (state_h, state_c))

            _, top_ix = torch.topk(output[0], k=top_k)
            choices = top_ix.tolist()
            choice = np.random.choice(choices[0])
            words.append(int_to_vocab[choice])

        print(' '.join(words))