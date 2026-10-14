"""Exercise the actual server wrapper without loading checkpoint weights."""
import types
import unittest
from unittest.mock import Mock

import numpy as np
import torch

from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server, PolicyPreprocessMixin


class RTCServingTest(unittest.TestCase):
    def server(self):
        server = LingbotVLAv2Server.__new__(LingbotVLAv2Server)
        server.config = types.SimpleNamespace(training_time_rtc=True,training_time_rtc_max_delay=10,
                                              n_action_steps=50,max_action_dim=55)
        server._infer_batch = Mock(return_value=(dict(action=np.zeros((1,50,14),np.float32)),
                                                torch.zeros(1,50,55)))
        server.chunk_ret=True
        server.use_length=8
        server.last_action_chunk={'action':np.full((1,8,14),99)}
        server.last_normalized_action_chunk=None
        server.global_step=1
        return server

    def test_rpc_forwards_full_normalized_prefix_without_truncating_or_caching(self):
        server=self.server()
        previous=np.ones((50,55),np.float32)
        request={'task':'test','_rtc':dict(request_id='episode:1',delay=6,previous_actions=previous)}
        response=server.infer(request)
        self.assertEqual(response['action'].shape,(50,14))
        self.assertEqual(response['_normalized_actions'].shape,(50,55))
        self.assertEqual(response['_rtc_request_id'],'episode:1')
        args, kwargs = server._infer_batch.call_args
        self.assertEqual(args,([{'task':'test'}],))
        self.assertEqual(kwargs['rtc_delay'],6)
        self.assertTrue(torch.equal(kwargs['previous_actions'],torch.ones(1,50,55)))
        self.assertTrue(kwargs['return_normalized'])
        server.infer(request)
        self.assertEqual(server._infer_batch.call_count,2)
        self.assertIn('_rtc',request)  # no mutation of caller's message

    def test_plain_request_retains_synchronous_chunk_slicing(self):
        server=self.server()
        server._infer_batch.return_value={'action':np.zeros((1,50,14),np.float32)}
        response=server.infer({'task':'test'})
        self.assertEqual(response['action'].shape,(8,14))
        self.assertNotIn('_normalized_actions',response)

    @unittest.skipUnless(torch.cuda.is_available(),'GPU needed for preprocessing device contract')
    def test_batch_wrapper_passes_rtc_parameters_to_model_sampler(self):
        policy=PolicyPreprocessMixin()
        policy.eval=lambda:None
        observations=dict(images=torch.zeros(1,3,3,2,2),img_masks=torch.ones(1,3,dtype=torch.bool),
                          lang_tokens=torch.zeros(1,2,dtype=torch.long),lang_masks=torch.ones(1,2,dtype=torch.bool),
                          state=torch.zeros(1,55))
        sampler=Mock(return_value=torch.zeros(1,50,55,device='cuda',dtype=torch.bfloat16))
        previous=torch.ones(1,50,55)
        result=policy.sample_actions_batch(observations,use_bf16=True,sample_compile_fn=sampler,
                                          previous_actions=previous,rtc_delay=6)
        kwargs=sampler.call_args.kwargs
        self.assertEqual(kwargs['rtc_delay'],6)
        self.assertEqual(kwargs['previous_actions'].device.type,'cuda')
        self.assertEqual(kwargs['previous_actions'].dtype,torch.bfloat16)
        self.assertEqual(result.device.type,'cpu')
        self.assertEqual(result.dtype,torch.float32)


if __name__=='__main__':
    unittest.main()
