"""System prompt wording of the standalone Hermes provider (#835)."""

from hermes_memory_provider import MnemosyneMemoryProvider


def test_system_prompt_keeps_always_injected_memory_files_as_a_tier():
    provider = MnemosyneMemoryProvider()
    provider._beam = object()

    block = provider.system_prompt_block()

    assert "Mnemosyne is primary" in block
    assert "deprecated for bulk durable storage" in block
    assert "always-injected memory files" in block
    assert "supported complementary tier" in block
