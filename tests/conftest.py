from typing import Callable, List

import pytest
from brownie import (
    BTCBurner,
    CBurner,
    Contract,
    ETHBurner,
    LPBurner,
    MetaBurner,
    UnderlyingBurner,
    USDNBurner,
    YBurner,
)
from brownie.network.account import Account, Accounts
from brownie.network.contract import ContractContainer
from brownie.network.state import Chain

YEAR = 365 * 86400
INITIAL_RATE = 274_815_283
YEAR_1_SUPPLY = INITIAL_RATE * 10 ** 18 // YEAR * YEAR
INITIAL_SUPPLY = 1_303_030_303


# helper functions as fixtures


def approx(a: int, b: int, precision: int = 1e-10) -> bool:
    """Verify whether a is approximately equal to b within the degree of precision."""
    if a == b == 0:
        return True
    return 2 * abs(a - b) / (a + b) <= precision


def pack_values(values: List[int]) -> bytes:
    """Helper function to pack integers into a bytes array of size 32."""
    packed = b"".join(i.to_bytes(1, "big") for i in values)
    padded = packed + bytes(32 - len(values))
    return padded


@pytest.fixture(autouse=True)
def isolation_setup(fn_isolation):
    """Isolate each function."""
    pass


@pytest.fixture(scope="function")
def theoretical_supply(chain: Chain, token: Contract) -> Callable:
    def _fn() -> int:
        epoch = token.mining_epoch()
        q = 1 / 2 ** 0.25
        S = INITIAL_SUPPLY * 10 ** 18
        if epoch > 0:
            S += int(YEAR_1_SUPPLY * (1 - q ** epoch) / (1 - q))
        S += int(YEAR_1_SUPPLY // YEAR * q ** epoch) * (
            chain[-1].timestamp - token.start_epoch_time()
        )
        return S

    return _fn


# account aliases


@pytest.fixture(scope="session")
def alice(accounts: Accounts) -> Account:
    """Return the first available test account."""
    return accounts[0]


@pytest.fixture(scope="session")
def bob(accounts: Accounts) -> Account:
    """Return the second available test account."""
    return accounts[1]


@pytest.fixture(scope="session")
def charlie(accounts: Accounts) -> Account:
    """Return the third available test account."""
    return accounts[2]


@pytest.fixture(scope="session")
def receiver(accounts: Accounts) -> Account:
    """Return a dummy receiver account."""
    return accounts.at("0x0000000000000000000000000000000000031337", True)


# core contracts


@pytest.fixture(scope="module")
def token(ERC20CRV: ContractContainer, alice: Account) -> Contract:
    """Return an instance of the ERC20CRV contract."""
    return ERC20CRV.deploy("Curve DAO Token", "CRV", 18, {"from": alice})


@pytest.fixture(scope="module")
def voting_escrow(VotingEscrow: ContractContainer, alice: Account, token: Contract) -> Contract:
    """Return an instance of the VotingEscrow contract."""
    return VotingEscrow.deploy(token, "Voting-escrowed CRV", "veCRV", "veCRV_0.99", {"from": alice})


@pytest.fixture(scope="module")
def gauge_controller(
    GaugeController: ContractContainer, alice: Account, token: Contract, voting_escrow: Contract,
) -> Contract:
    """Return an instance of the GaugeController contract."""
    return GaugeController.deploy(token, voting_escrow, {"from": alice})


@pytest.fixture(scope="module")
def minter(
    Minter: ContractContainer, alice: Account, gauge_controller: Contract, token: Contract,
) -> Contract:
    """Return an instance of the Minter contract."""
    return Minter.deploy(token, gauge_controller, {"from": alice})


@pytest.fixture(scope="module")
def pool_proxy(PoolProxy: ContractContainer, alice: Account) -> Contract:
    """Return an instance of the PoolProxy contract."""
    return PoolProxy.deploy(alice, alice, alice, {"from": alice})


@pytest.fixture(scope="module")
def gauge_proxy(GaugeProxy: ContractContainer, alice: Account, bob: Account) -> Contract:
    """Return an instance of the GaugeProxy contract."""
    return GaugeProxy.deploy(alice, bob, {"from": alice})


@pytest.fixture(scope="module")
def coin_reward(ERC20: ContractContainer, alice: Account) -> Contract:
    """Return a mock ERC20 contract."""
    return ERC20.deploy("YFIIIIII Funance", "YFIIIIII", 18, {"from": alice})


@pytest.fixture(scope="module")
def reward_contract(
    CurveRewards: ContractContainer, mock_lp_token: Contract, alice: Account, coin_reward: Contract,
):
    """Return an instance of the CurveRewards contract."""
    contract = CurveRewards.deploy(mock_lp_token, coin_reward, {"from": alice})
    contract.setRewardDistribution(alice, {"from": alice})
    return contract


@pytest.fixture(scope="module")
def liquidity_gauge(
    LiquidityGauge: ContractContainer, alice: Account, mock_lp_token: Contract, minter: Contract
) -> Contract:
    """Return an instance of the LiquidityGauge contract."""
    return LiquidityGauge.deploy(mock_lp_token, minter, alice, {"from": alice})


@pytest.fixture(scope="module")
def gauge_v2(
    LiquidityGaugeV2: ContractContainer, alice: Account, mock_lp_token: Contract, minter: Contract
) -> Contract:
    """Return an instance of the LiquidityGaugeV2 contract."""
    return LiquidityGaugeV2.deploy(mock_lp_token, minter, alice, {"from": alice})


@pytest.fixture(scope="module")
def gauge_wrapper(
    LiquidityGaugeWrapper: ContractContainer, alice: Account, liquidity_gauge: Contract
) -> Contract:
    """Return an instance of the LiquidityGaugeWrapper contract."""
    return LiquidityGaugeWrapper.deploy(
        "Tokenized Gauge", "TG", liquidity_gauge, alice, {"from": alice}
    )


@pytest.fixture(scope="module")
def liquidity_gauge_reward(
    LiquidityGaugeReward: ContractContainer,
    alice: Account,
    mock_lp_token: Contract,
    minter: Contract,
    reward_contract: Contract,
    coin_reward: Contract,
) -> Contract:
    """Return an instance of the LiquidityGaugeReward contract."""
    return LiquidityGaugeReward.deploy(
        mock_lp_token, minter, reward_contract, coin_reward, alice, {"from": alice},
    )


@pytest.fixture(scope="module")
def reward_gauge_wrapper(
    LiquidityGaugeRewardWrapper: ContractContainer,
    alice: Account,
    liquidity_gauge_reward: Contract,
) -> Contract:
    """Return an instance of the LiquidityGaugeRewardWrapper contract."""
    return LiquidityGaugeRewardWrapper.deploy(
        "Tokenized Reward Gauge", "TG", liquidity_gauge_reward, alice, {"from": alice},
    )


@pytest.fixture(scope="module")
def three_gauges(
    LiquidityGauge: ContractContainer, alice: Account, mock_lp_token: Contract, minter: Contract
) -> List[Contract]:
    """Return a list of three LiquidityGauge contracts."""
    contracts = [
        LiquidityGauge.deploy(mock_lp_token, minter, alice, {"from": alice}) for _ in range(3)
    ]

    return contracts


# VestingEscrow fixtures


@pytest.fixture(scope="module")
def start_time(chain: Chain) -> int:
    """Return the timestamp of a year + approx. sixteen minutes from the chain time."""
    return chain.time() + 1000 + 86400 * 365


@pytest.fixture(scope="module")
def end_time(start_time: int):
    """Return the timestamp of start_time + approx. 3 years."""
    return start_time + 100_000_000


@pytest.fixture(scope="module")
def vesting(
    VestingEscrow: ContractContainer,
    alice: Account,
    accounts: Accounts,
    coin_a: Contract,
    start_time: int,
    end_time: int,
) -> Contract:
    """Return an instance of the VestingEscrow contract."""
    contract = VestingEscrow.deploy(
        coin_a, start_time, end_time, True, accounts[1:5], {"from": alice}
    )
    coin_a._mint_for_testing(10 ** 21, {"from": alice})
    coin_a.approve(contract, 10 ** 21, {"from": alice})
    return contract


@pytest.fixture(scope="module")
def vesting_target(VestingEscrowSimple: ContractContainer, alice: Account) -> Contract:
    """Return an instance of the VestingEscrowSimple contract."""
    return VestingEscrowSimple.deploy({"from": alice})


@pytest.fixture(scope="module")
def vesting_factory(
    VestingEscrowFactory: ContractContainer, alice: Account, vesting_target: Contract
) -> Contract:
    """Return an instance of the VestingEscrowFactory contract."""
    return VestingEscrowFactory.deploy(vesting_target, alice, {"from": alice})


@pytest.fixture(scope="module")
def vesting_simple(
    VestingEscrowSimple: ContractContainer,
    alice: Account,
    bob: Account,
    vesting_factory: Contract,
    coin_a: Contract,
    start_time: int,
) -> Contract:
    """Deploy a VestingEscrow contract through VestingEscrowFactory and return it."""
    coin_a._mint_for_testing(10 ** 21, {"from": alice})
    coin_a.transfer(vesting_factory, 10 ** 21, {"from": alice})
    tx = vesting_factory.deploy_vesting_contract(
        coin_a, bob, 10 ** 20, True, 100000000, start_time, {"from": alice},
    )
    return VestingEscrowSimple.at(tx.new_contracts[0])


# parametrized burner fixture


@pytest.fixture(
    scope="module",
    params=[
        BTCBurner,
        CBurner,
        ETHBurner,
        LPBurner,
        MetaBurner,
        UnderlyingBurner,
        USDNBurner,
        YBurner,
    ],
)
def burner(
    alice: Account, bob: Account, receiver: Account, pool_proxy: Contract, request
) -> Contract:
    """Parameterized fixture which returns a Burner contract."""
    Burner: ContractContainer = request.param
    args = (pool_proxy, receiver, receiver, alice, bob, {"from": alice})
    idx = len(Burner.deploy.abi["inputs"]) + 1

    return Burner.deploy(*args[-idx:])

    # if len(Burner.deploy.abi['inputs']) == 4:
    #     return Burner.deploy(receiver, receiver, alice, bob, {'from': alice})
    # else:
    #     return Burner.deploy(receiver, alice, bob, {'from': alice})


# testing contracts


@pytest.fixture(scope="module")
def coin_a(ERC20: ContractContainer, alice: Account) -> Contract:
    """Return a mock ERC20 contract."""
    return ERC20.deploy("Coin A", "USDA", 18, {"from": alice})


@pytest.fixture(scope="module")
def coin_b(ERC20: ContractContainer, alice: Account) -> Contract:
    """Return a mock ERC20 contract."""
    return ERC20.deploy("Coin B", "USDB", 18, {"from": alice})


@pytest.fixture(scope="module")
def mock_lp_token(ERC20LP: ContractContainer, alice: Account) -> Contract:
    """Return a mock LP ERC20 contract.

    Note:
        Not using the actual Curve LP contract.
    """
    return ERC20LP.deploy("Curve LP token", "usdCrv", 18, 10 ** 9, {"from": alice})


@pytest.fixture(scope="module")
def pool(
    CurvePool: ContractContainer,
    alice: Account,
    mock_lp_token: Contract,
    coin_a: Contract,
    coin_b: Contract,
) -> Contract:
    """Return a CurvePool contract"""
    curve_pool = CurvePool.deploy(
        [coin_a, coin_b], mock_lp_token, 100, 4 * 10 ** 6, {"from": alice}
    )
    mock_lp_token.set_minter(curve_pool, {"from": alice})

    return curve_pool


@pytest.fixture(scope="module")
def fee_distributor(
    FeeDistributor: ContractContainer,
    voting_escrow: Contract,
    alice: Account,
    coin_a: Contract,
    chain: Chain,
) -> Callable:
    """Return a callable which deploys a FeeDistributor contract."""

    def f(t: int = None) -> Contract:
        if not t:
            t = chain.time()
        return FeeDistributor.deploy(voting_escrow, t, coin_a, alice, alice, {"from": alice})

    return f
