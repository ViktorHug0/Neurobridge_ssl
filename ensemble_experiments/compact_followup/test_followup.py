import unittest
import torch
from ensemble_experiments.compact_valcon.models import CompactDecoder
from .models import make_model, distillation_loss


class FollowupTests(unittest.TestCase):
    def test_electrode_shapes_gradients_and_sharing(self):
        model=make_model('electrode');calls=[]
        handle=model.backbone.tsconv[0].register_forward_hook(lambda *args:calls.append(1))
        before=[n.num_batches_tracked.item() for n in model.spatial_norms]
        features=model(torch.randn(9,63,250),torch.randn(9,6400))
        self.assertEqual(calls,[1]);handle.remove()
        for e,i in features:
            self.assertEqual(tuple(e.shape),(9,128));self.assertEqual(tuple(i.shape),(9,128))
        loss=model.loss(features,torch.eye(9,dtype=torch.bool));loss.backward()
        for name,p in model.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad,name);self.assertTrue(torch.isfinite(p.grad).all(),name)
        self.assertEqual([n.num_batches_tracked.item()-v for n,v in zip(model.spatial_norms,before)],[1,1])
        self.assertLess(model.size()['parameters'],CompactDecoder('single').size()['parameters']*1.05)

    def test_distill_student_identical_to_previous(self):
        torch.manual_seed(3300);a=make_model('distill')
        torch.manual_seed(3300);b=CompactDecoder('dual_head')
        for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)

    def test_kd_teacher_detached_and_self_loss_zero(self):
        student=[(torch.randn(9,128,requires_grad=True),torch.randn(9,128,requires_grad=True)) for _ in range(2)]
        teacher=[(torch.randn(9,128,requires_grad=True),torch.randn(9,128,requires_grad=True)) for _ in range(2)]
        objects=torch.arange(9)//3;ids=torch.zeros(9,dtype=torch.long)
        loss=distillation_loss(student,teacher,objects,ids);loss.backward()
        for e,i in teacher:self.assertIsNone(e.grad);self.assertIsNone(i.grad)
        for e,i in student:self.assertIsNotNone(e.grad);self.assertIsNotNone(i.grad)
        self.assertLess(abs(distillation_loss(student,student,objects,ids).item()),1e-6)

    def test_checkpoint_roundtrip(self):
        model=make_model('electrode').eval();other=make_model('electrode').eval()
        other.load_state_dict(model.state_dict());eeg=torch.randn(2,63,250)
        with torch.no_grad():
            for a,b in zip(model.encode(eeg),other.encode(eeg)):torch.testing.assert_close(a,b,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
