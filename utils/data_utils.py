
def bio2bmes(slots):
    chunks = get_chunks(slots + ['O'])
    bmes_slots = ['O'] * len(slots)
    for chunk in chunks:
        if chunk[0] == chunk[1]:
            bmes_slots[chunk[0]] = 'S-' + chunk[2]
        else:
            bmes_slots[chunk[0]] = 'B-' + chunk[2]
            for idx in range(chunk[0] + 1, chunk[1]):
                bmes_slots[idx] = 'M-' + chunk[2]
            bmes_slots[chunk[1]] = "E-" + chunk[2]
    return bmes_slots

def get_chunks(labels):
    """
        It supports IOB2 or IOBES tagging scheme.
        You may also want to try https://github.com/sighsmile/conlleval.
    """
    chunks = []
    start_idx,end_idx = 0,0
    for idx in range(1,len(labels)-1):
        chunkStart, chunkEnd = False,False
        if labels[idx-1] not in ('O', '<pad>', '<unk>', '<s>', '</s>', '<STOP>', '<START>'):
            prevTag, prevType = labels[idx-1][:1], labels[idx-1][2:]
        else:
            prevTag, prevType = 'O', 'O'
        if labels[idx] not in ('O', '<pad>', '<unk>', '<s>', '</s>', '<STOP>', '<START>'):
            Tag, Type = labels[idx][:1], labels[idx][2:]
        else:
            Tag, Type = 'O', 'O'
        if labels[idx+1] not in ('O', '<pad>', '<unk>', '<s>', '</s>', '<STOP>', '<START>'):
            nextTag, nextType = labels[idx+1][:1], labels[idx+1][2:]
        else:
            nextTag, nextType = 'O', 'O'

        if Tag == 'B' or Tag == 'S' or (prevTag, Tag) in {('O', 'I'), ('O', 'E'), ('E', 'I'), ('E', 'E'), ('S', 'I'), ('S', 'E')}:
            chunkStart = True
        if Tag != 'O' and prevType != Type:
            chunkStart = True

        if Tag == 'E' or Tag == 'S' or (Tag, nextTag) in {('B', 'B'), ('B', 'O'), ('B', 'S'), ('I', 'B'), ('I', 'O'), ('I', 'S')}:
            chunkEnd = True
        if Tag != 'O' and Type != nextType:
            chunkEnd = True

        if chunkStart:
            start_idx = idx
        if chunkEnd:
            end_idx = idx
            chunks.append((start_idx,end_idx,Type))
            start_idx,end_idx = 0,0
    return chunks