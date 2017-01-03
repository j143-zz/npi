
import lstm

seq_len = self.seq_len
        num_embed = self.num_embed
        num_lstm_layer = self.num_lstm_layer
        num_hidden = self.num_hidden

        f_lstm = lstm.lstm_unroll(
            num_lstm_layer,
            seq_len,
            len(vocab) + 1,
            num_hidden=num_hidden
            num_embed=num_embed,
            num_label=len(vocab) + 1,
            dropout = 0.1
        )
