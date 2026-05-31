# EMGIO Development Plan

## Project Overview
**Goal:** Unified EMG data import/export library with metadata preservation
**Timeline:** Ongoing maintenance and enhancement
**Stack:** Python 3.11+, numpy, pandas, scipy, matplotlib, pyedflib, wfdb, pyxdf (managed with UV)

## Development Tasks

### Phase 1: Foundation [COMPLETED]
- [x] Setup project structure and configuration
- [x] Create Python package skeleton
- [x] Configure build and dependency management  
- [x] Add BSD 3-Clause licensing
- [x] Setup GitHub Actions CI/CD

### Phase 2: Core Features [COMPLETED]
- [x] Implement Recording class with signal handling
- [x] Add multi-format importers (EEGLAB, Trigno, OTB, EDF/BDF, WFDB, XDF, CSV)
- [x] Create EDF/BDF exporter with auto format selection
- [x] Implement metadata and channel management
- [x] Add error handling and validation

### Phase 3: Integration & Polish [IN PROGRESS]
- [x] Add WFDB integration with annotations
- [ ] Enhance EDF+/BDF+ annotation support
- [x] Create comprehensive testing suite
- [ ] Add performance optimizations for large files
- [ ] Implement Noraxon importer

### Phase 4: Documentation & Release [ONGOING]
- [x] Create MkDocs documentation site
- [x] Add usage examples
- [ ] Add more tutorials for each format
- [x] Setup automated testing
- [x] Publish to PyPI (current release 0.2.2)

## Success Criteria
- [x] All major EMG formats supported
- [x] Documentation site live
- [x] CI/CD pipeline functional
- [ ] 90%+ test coverage achieved
- [ ] Round-trip conversion validated

## Current Focus
- Improving annotation handling across formats
- Expanding test coverage with real data
- Optimizing memory usage for large datasets

## Notes
- Signal integrity is paramount - no data loss during conversion
- All features must be tested with real EMG data
- Automatic format detection should be reliable 