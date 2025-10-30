from typing import List


from lib.agents.claim_verifier import EvidenceAlignmentLevel
from lib.agents.evidence_weighter import EvidenceWeighterRecommendedAction
from lib.agents.models import ClaimCategory
from lib.workflows.claim_substantiation.state import (
    ClaimSubstantiatorState,
    ContextData,
    DocumentChunk,
    DocumentIssue,
    SeverityEnum,
)


def rank_issues(state: ClaimSubstantiatorState) -> ClaimSubstantiatorState:
    """Rank issues based on analysis results from various workflow stages."""
    issues: List[DocumentIssue] = []

    # 1. Extract References: References without matching supporting documents
    for reference in state.references:
        if not reference.has_associated_supporting_document:
            issue = DocumentIssue(
                title="Missing supporting document for reference",
                description=f'Reference does not have an associated supporting document: "{reference.text}"',
                severity=SeverityEnum.LOW,
                additional_context="This reference was mentioned in the document but no corresponding supporting document was provided. It was excluded from the analysis.",
                chunk_index=_find_chunk_index_by_text(state, reference.text),
            )
            issues.append(issue)

    # 2. Reference Validation: Invalid references
    for validation in state.references_validated:
        if not validation.valid_reference:
            issue = DocumentIssue(
                title="Invalid reference",
                description=f'Possible invalid reference: "{validation.original_reference.text}"',
                severity=SeverityEnum.HIGH,
                chunk_index=_find_chunk_index_by_text(
                    state, validation.original_reference.text
                ),
                context_data=[
                    ContextData(
                        label="Suggested action",
                        value=validation.suggested_action,
                    ),
                ]
                + [
                    ContextData(
                        label=field_validation.category.value.capitalize(),
                        value=[
                            ContextData(
                                label="Issue",
                                value=field_validation.problem_type.value,
                            ),
                            ContextData(
                                label="Current value",
                                value=field_validation.current_value,
                            ),
                            ContextData(
                                label="Suggested value",
                                value=field_validation.suggested_value,
                            ),
                        ],
                    )
                    for field_validation in validation.bibliography_field_validations
                ],
            )
            issues.append(issue)

    # 3. Claim Categorization: Claims needing external verification without citations
    for chunk in state.chunks:
        if not chunk.claim_categories:
            continue

        # Check if chunk has citations
        has_citations = (
            chunk.citations
            and chunk.citations.citations
            and len(chunk.citations.citations) > 0
        )

        for category in chunk.claim_categories:
            if category.needs_external_verification and not has_citations:
                issue = DocumentIssue(
                    title="Unsupported claim",
                    description=f"Claim '{category.claim}' requires external verification but no citations were found.",
                    severity=SeverityEnum.MEDIUM,
                    chunk_index=category.chunk_index,
                    claim_index=category.claim_index,
                    claim_category=category.claim_category,
                    context_data=[
                        ContextData(
                            label="Rationale",
                            value=category.rationale,
                        ),
                        ContextData(
                            label="Needs external verification",
                            value=category.needs_external_verification,
                        ),
                    ],
                )
                issues.append(issue)

    # 4. Claim Verification: Unsupported and partially supported claims
    for chunk in state.chunks:
        if not chunk.substantiations:
            continue

        for substantiation in chunk.substantiations:
            if substantiation.evidence_alignment in [
                EvidenceAlignmentLevel.UNSUPPORTED,
                EvidenceAlignmentLevel.PARTIALLY_SUPPORTED,
                EvidenceAlignmentLevel.UNVERIFIABLE,
            ]:
                title = (
                    "Unsupported Claim"
                    if substantiation.evidence_alignment
                    == EvidenceAlignmentLevel.UNSUPPORTED
                    else (
                        "Partially Supported Claim"
                        if substantiation.evidence_alignment
                        == EvidenceAlignmentLevel.PARTIALLY_SUPPORTED
                        else "Unverifiable Claim"
                    )
                )
                issue = DocumentIssue(
                    title=title,
                    description=substantiation.rationale,
                    severity=(
                        SeverityEnum.HIGH
                        if substantiation.evidence_alignment
                        == EvidenceAlignmentLevel.UNSUPPORTED
                        else SeverityEnum.MEDIUM
                    ),
                    chunk_index=substantiation.chunk_index,
                    claim_index=substantiation.claim_index,
                    claim_category=_find_claim_category(
                        chunk, substantiation.claim_index
                    ),
                    context_data=[
                        ContextData(
                            label="Evidence alignment",
                            value=substantiation.evidence_alignment.value,
                        ),
                        ContextData(
                            label="Feedback",
                            value=substantiation.feedback,
                        ),
                    ]
                    + [
                        ContextData(
                            label=f"Evidence source {index + 1}",
                            value=[
                                ContextData(
                                    label="Reference file name",
                                    value=source.reference_file_name,
                                ),
                                ContextData(
                                    label="Location",
                                    value=source.location,
                                ),
                                ContextData(
                                    label="Quote",
                                    value=source.quote,
                                ),
                            ],
                        )
                        for index, source in enumerate(substantiation.evidence_sources)
                    ],
                )
                issues.append(issue)

    # 5. Live Reports Analysis: Add citation and update claim actions
    for live_report in state.live_reports_analysis:
        if (
            live_report.claim_update_action
            == EvidenceWeighterRecommendedAction.ADD_CITATION
        ):
            issue = DocumentIssue(
                title="Additional Citation Recommended",
                description=live_report.rationale,
                severity=SeverityEnum.MEDIUM,
                additional_context=f"Newer references alignment: {live_report.newer_references_alignment.value}. Confidence: {live_report.confidence_level.value}.",
                chunk_index=live_report.chunk_index,
                claim_index=live_report.claim_index,
                claim_category=_find_claim_category(chunk, live_report.claim_index),
            )
            issues.append(issue)
        elif (
            live_report.claim_update_action
            == EvidenceWeighterRecommendedAction.UPDATE_CLAIM
        ):
            issue = DocumentIssue(
                title="Claim Update Recommended",
                description=live_report.rationale,
                severity=SeverityEnum.MEDIUM,
                additional_context=f"Newer references alignment: {live_report.newer_references_alignment.value}. Confidence: {live_report.confidence_level.value}. Rewritten claim: {live_report.rewritten_claim}",
                chunk_index=live_report.chunk_index,
                claim_index=live_report.claim_index,
                claim_category=_find_claim_category(chunk, live_report.claim_index),
            )
            issues.append(issue)

    issues.sort(key=lambda x: x.severity.sort_index(), reverse=True)

    return {"ranked_issues": issues}


def _find_claim_category(chunk: DocumentChunk, claim_index: int) -> ClaimCategory:
    for category in chunk.claim_categories:
        if category.claim_index == claim_index:
            return category.claim_category

    return None


def _find_chunk_index_by_text(state: ClaimSubstantiatorState, text: str) -> int:
    for chunk in state.chunks:
        if text in chunk.content:
            return chunk.chunk_index

    return None
