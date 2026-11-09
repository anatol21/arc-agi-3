"""Transformers inference with durable chunk callbacks and a process-local model cache."""
def generate_chunks(model, input_ids, attention_mask, *, generation_config, row_limits,
                    chunk_tokens, stop, commit, save_on_terminal=False, allow_sampling=False,
                    **generation_kwargs):
    """Keep one generate call alive. commit(rows) receives new token IDs per original row.

    Caller owns the model/adapter context and must atomically persist each delta before returning.
    A restarted worker reconstructs its input from the journal. No GPU state is serialized.
    """
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList
    if generation_config.do_sample and not allow_sampling:
        raise ValueError("Chunk streaming requires explicit opt-in for stochastic sampling")
    if (generation_config.num_beams != 1 or generation_config.num_return_sequences != 1
            or not generation_config.use_cache):
        raise ValueError("Chunk streaming requires single-output cached generation")
    eos, pad = generation_config.eos_token_id, generation_config.pad_token_id
    if not isinstance(eos,int) or not isinstance(pad,int) or eos == pad:
        raise ValueError("Use a single EOS token and distinct padding token")
    if (input_ids.shape != attention_mask.shape or len(row_limits) != len(input_ids)
            or not row_limits or any(type(n) != int or n < 1 for n in row_limits)
            or type(chunk_tokens) != int or chunk_tokens < 1
            or generation_config.max_new_tokens != max(row_limits)):
        raise ValueError("Invalid batch dimensions or per-row generation budgets")
    if stop():
        return
    width = input_ids.shape[1]
    limits = torch.tensor(row_limits,device=input_ids.device)
    ended = torch.zeros_like(limits,dtype=torch.bool)
    lengths = torch.zeros_like(limits)

    class Checkpoint(StoppingCriteria):
        saved = 0

        def __call__(self, ids, scores, **kwargs):
            count = ids.shape[1] - width
            lengths.add_((~ended).long())
            finished_now = (~ended) & ((ids[:,-1] == eos) | (count >= limits))
            ended.logical_or_(finished_now)
            interrupted = stop()
            if count % chunk_tokens == 0 or interrupted or (save_on_terminal and finished_now.any().item()):
                self.save(ids)
            return torch.ones_like(ended) if interrupted else ended.clone()

        def save(self, ids):
            count = ids.shape[1] - width
            if self.saved == count:
                return
            emitted = ids[:,width+self.saved:width+count].tolist()
            delta = [row[:max(0,n-self.saved)] for row,n in zip(emitted,lengths.tolist(),strict=True)]
            if any(delta):
                commit(delta)
            self.saved = count

    checkpoint = Checkpoint()
    with torch.inference_mode():
        result = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                       generation_config=generation_config,
                       stopping_criteria=StoppingCriteriaList([checkpoint]), **generation_kwargs)
        checkpoint.save(result.sequences if hasattr(result,"sequences") else result)
