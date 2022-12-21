# Owner(s): ["oncall: quantization"]
import torch
import torch._dynamo as torchdynamo
from torch.testing._internal.common_quantization import (
    QuantizationTestCase,
    skip_if_no_torchvision,
    skipIfNoQNNPACK,
)
from torch.testing._internal.common_quantized import (
    override_quantized_engine,
)
from torch.ao.quantization import (
    get_default_qconfig,
    QConfigMapping,
)
from torch.ao.quantization.backend_config import (
    get_qnnpack_backend_config,
)
from torch.ao.quantization.backend_config._qnnpack_pt2e import get_qnnpack_pt2e_backend_config
from torch.ao.quantization.backend_config._inductor_pt2e import get_inductor_pt2e_backend_config
from torch.ao.quantization.quantize_fx import prepare_fx, convert_to_reference_fx
from torch.ao.quantization._quantize_pt2e import prepare_pt2e, convert_pt2e
from torch.ao.ns.fx.utils import (
    compute_sqnr,
)
from torch._inductor.compile_fx import compile_fx

class TestQuantizePT2EModels(QuantizationTestCase):
    @skip_if_no_torchvision
    @skipIfNoQNNPACK
    def test_resnet18(self):
        import copy
        import torchvision
        with override_quantized_engine("qnnpack"):
            example_inputs = (torch.randn(1, 3, 224, 224),)
            m = torchvision.models.resnet18().eval()
            m_copy = copy.deepcopy(m)
            # program capture
            m, guards = torchdynamo.export(
                m,
                *copy.deepcopy(example_inputs),
                aten_graph=True,
                tracing_mode="real",
            )

            backend_config = get_qnnpack_pt2e_backend_config()
            # TODO: define qconfig_mapping specifically for executorch
            qconfig = get_default_qconfig("qnnpack")
            qconfig_mapping = QConfigMapping().set_global(qconfig)
            before_fusion_result = m(*example_inputs)

            m = prepare_pt2e(m, qconfig_mapping, example_inputs, backend_config)

            # checking that we inserted observers correctly for maxpool operator (input and
            # output share observer instance)
            self.assertEqual(id(m.activation_post_process_3), id(m.activation_post_process_2))
            after_prepare_result = m(*example_inputs)
            m = convert_pt2e(m)

            after_quant_result = m(*example_inputs)

            # comparing with existing fx graph mode quantization reference flow
            backend_config = get_qnnpack_backend_config()
            m_fx = prepare_fx(m_copy, qconfig_mapping, example_inputs, backend_config=backend_config)
            after_prepare_result_fx = m_fx(*example_inputs)
            m_fx = convert_to_reference_fx(m_fx, backend_config=backend_config)

            after_quant_result_fx = m_fx(*example_inputs)

            # the result matches exactly after prepare
            self.assertEqual(after_prepare_result, after_prepare_result_fx)
            self.assertEqual(compute_sqnr(after_prepare_result, after_prepare_result_fx), torch.tensor(float("inf")))
            # there are slight differences after convert due to different implementations
            # of quant/dequant
            self.assertTrue(torch.max(after_quant_result - after_quant_result_fx) < 1e-1)
            self.assertTrue(compute_sqnr(after_quant_result, after_quant_result_fx) > 35)

    def test_inductor_backend(self):
        '''
        Inductor as a quantization backend. For experiment.
        '''
        import copy
        from torch import _dynamo, _inductor
        from torch._inductor import config
        import logging

        torch._dynamo.config.log_level = logging.DEBUG
        torch._dynamo.config.verbose = True
        torch._inductor.config.trace.enabled = True
        torch._inductor.config.debug = True

        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(x + x)

        with override_quantized_engine("x86"):
            example_inputs = (torch.randn(1, 3, 224, 224),)
            m = Mod().eval()
            # program capture
            m, guards = torchdynamo.export(
                m,
                *copy.deepcopy(example_inputs),
                aten_graph=True,
                tracing_mode="real",
            )

            backend_config = get_inductor_pt2e_backend_config()
            qconfig = get_default_qconfig("x86")
            qconfig_mapping = QConfigMapping().set_global(qconfig)
            before_fusion_result = m(*example_inputs)

            m = prepare_pt2e(m, qconfig_mapping, example_inputs, backend_config)
            after_prepare_result = m(*example_inputs)

            m = convert_pt2e(m)
            after_quant_result = m(*example_inputs)

            # A few ops in EXIR are not supported. Set nopython=False to make it work
            run = torch._dynamo.optimize(compile_fx, nopython=False)(m)

            # first run
            inductor_result = run(*example_inputs)

            module_result = m(*example_inputs)
            self.assertEqual(inductor_result, module_result)

            # second run
            inductor_result = run(*example_inputs)

