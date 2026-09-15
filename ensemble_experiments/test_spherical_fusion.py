import unittest
import numpy as np
from ensemble_experiments.spherical_fusion import procrustes, scores, slerp, unit, validation


class GeometryTests(unittest.TestCase):
    def test_rotation_and_fusion(self):
        rng = np.random.default_rng(7)
        a = unit(rng.normal(size=(300, 16)))
        r, _ = np.linalg.qr(rng.normal(size=(16,16)))
        b = a @ r
        fit, _ = procrustes(a,b)
        np.testing.assert_allclose(b@fit, a, atol=1e-12)
        q = np.stack([a[:20],b[:20]])
        v = np.stack([a[20:40],b[20:40]])
        result = scores(q,v,fit)
        for value in result.values(): self.assertTrue(np.isfinite(value).all())
        np.testing.assert_allclose(result['slerp_0.5'],result['solo_ts'],atol=1e-12)
        # Arbitrary independent rotations cannot affect aligned fusion.
        other, _ = np.linalg.qr(rng.normal(size=(16,16)))
        fit2, _ = procrustes(a,b@other)
        rotated = scores(np.stack([a[:20],b[:20]@other]),
                         np.stack([a[20:40],b[20:40]@other]),fit2)
        for k in result: np.testing.assert_allclose(result[k],rotated[k],atol=1e-11)

    def test_midpoint_and_expansion(self):
        rng = np.random.default_rng(9)
        a,b,x,y = [unit(rng.normal(size=(20,16))) for _ in range(4)]
        np.testing.assert_allclose(slerp(a,b,.5),unit(a+b),atol=1e-12)
        result = scores(np.stack([a,b]),np.stack([x,y]),np.eye(16))
        denominator = np.linalg.norm(a+b,axis=1)[:,None]*np.linalg.norm(x+y,axis=1)[None,:]
        np.testing.assert_allclose(result['four_terms']*4/denominator,
                                   result['slerp_0.5'],atol=1e-12)

    def test_validation_coverage(self):
        ids = np.tile(np.arange(1650),9)
        q = np.broadcast_to(ids[None,:,None],(2,len(ids),1))
        v = np.broadcast_to(np.arange(1650)[None,:,None],(2,1650,1))
        count = 0
        for qp,vp,target in validation(q,v,ids):
            self.assertEqual(vp.shape[1],200)
            self.assertEqual(len(np.unique(vp[0,:,0])),200)
            np.testing.assert_array_equal(qp[0,:,0],vp[0,target,0])
            count += len(target)
        self.assertEqual(count,len(ids))


if __name__ == '__main__': unittest.main()
