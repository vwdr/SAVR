"""Synthetic P0--P4 episode/reset state semantics for B2."""

from __future__ import annotations

from dataclasses import dataclass

from savr.brace.types import B2ValidationError, ProfileFamily


@dataclass(frozen=True)
class RuntimeTransition:
    family: ProfileFamily
    query: int
    dense_executed: bool
    accelerated_executed: bool
    action_discarded: bool
    cache_source_query: int | None
    gate_source_query: int | None
    pending_dense_query: int | None


@dataclass(frozen=True)
class ProfileRuntime:
    family: ProfileFamily
    episode_id: str
    last_query: int = -1
    cache_source_query: int | None = None
    gate_source_query: int | None = None
    pending_dense_query: int | None = None

    @classmethod
    def reset(cls, family: ProfileFamily, episode_id: str) -> "ProfileRuntime":
        if not episode_id:
            raise B2ValidationError("runtime reset requires an episode identity")
        return cls(family=family, episode_id=episode_id)

    def step(self, query: int, *, contract_active: bool = False) -> tuple["ProfileRuntime", RuntimeTransition]:
        if query != self.last_query + 1:
            raise B2ValidationError("profile runtime queries must be contiguous")
        dense = accelerated = discarded = False
        cache_source = self.cache_source_query
        gate_source = self.gate_source_query
        pending = self.pending_dense_query

        if self.family is ProfileFamily.P0:
            dense = True
            cache_source = gate_source = query
        elif self.family in (ProfileFamily.P1, ProfileFamily.P2):
            if contract_active:
                if cache_source is None or gate_source is None:
                    raise B2ValidationError("P1/P2 acceleration lacks a dense anchor")
                accelerated = True
            else:
                dense = True
                cache_source = gate_source = query
        elif self.family is ProfileFamily.P3:
            if cache_source is None:
                dense = True
            else:
                accelerated = True
            cache_source = gate_source = query
        elif self.family is ProfileFamily.P4:
            dense = True
            discarded = True
            if pending is not None:
                accelerated = True
                cache_source = gate_source = pending
            pending = query
        else:
            raise B2ValidationError("unknown profile family")

        next_state = ProfileRuntime(
            family=self.family,
            episode_id=self.episode_id,
            last_query=query,
            cache_source_query=cache_source,
            gate_source_query=gate_source,
            pending_dense_query=pending,
        )
        transition = RuntimeTransition(
            family=self.family,
            query=query,
            dense_executed=dense,
            accelerated_executed=accelerated,
            action_discarded=discarded,
            cache_source_query=cache_source,
            gate_source_query=gate_source,
            pending_dense_query=pending,
        )
        return next_state, transition
