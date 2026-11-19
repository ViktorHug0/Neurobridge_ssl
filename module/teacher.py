"""Frozen teacher ensemble for distillation (train.py --kd_teacher; scripts/subject_key/KEY_LOG.md).
The file comes from scripts/subject_key/export_teacher.py: per member, the L2-normalised embedding of every clean
training trial (subject x image) and of every training image. The ensemble score is the mean of the members'
EEG-image cosines, as in the inductive score ensembles."""
import torch


class Teachers:
    def __init__(self, path, args, device):
        t = torch.load(path, map_location=device)
        assert sorted(t['subjects']) == sorted(args.train_subject_ids), path
        self.eeg, self.img = t['eeg'], t['img']  # per member: (subjects * images, d), (images, d), float16
        self.n_img = len(self.img[0])
        self.row = torch.full((max(t['subjects']) + 1,), -1, device=device, dtype=torch.long)
        self.row[torch.tensor(t['subjects'], device=device)] = torch.arange(len(t['subjects']), device=device)

    @torch.no_grad()
    def cosine(self, subject_ids, flat, member=None):
        """(B, B) cos(trial i, image of trial j) of one member, or the mean over the members; flat = object * 10 + image index."""
        rows = self.row[subject_ids] * self.n_img + flat
        pairs = list(zip(self.eeg, self.img)) if member is None else [(self.eeg[member], self.img[member])]
        return sum(e[rows].float() @ g[flat].float().T for e, g in pairs) / len(pairs)
