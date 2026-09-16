# Research Notes

## Project Context

This platform integrates crowdsourced acoustic sensing, deep-learning sound classification, event-driven streaming, spatial-temporal storage, H3 spatial aggregation, context-aware anomaly detection, and real-time visualization into one experimental platform.

The research value comes from investigating how these components work together and what limitations appear.

## Key References

- Salamon, J., Jacoby, C., & Bello, J. P. (2014). A Dataset and Taxonomy for Urban Sound Research. ACM International Conference on Multimedia.
- UrbanSound8K dataset: https://urbansounddataset.weebly.com/urbansound8k.html
- H3: Uber's Hexagonal Hierarchical Spatial Index: https://h3geo.org/

## Known Limitations

- UrbanSound8K is not a perfect representation of every real urban environment
- Dataset recordings may differ from smartphone recordings
- Simulated GPS is not real GPS
- Simulated decibel values are not professional SPL measurements
- Smartphone microphones have different frequency responses
- Background noise can cause classification errors
- Sparse sensor coverage creates uncertainty
- H3 aggregation simplifies spatial variation
- Anomaly detection depends heavily on the quality of the historical baseline
- Kafka simulation does not prove city-scale scalability
- Local experiments do not represent production performance

---

*Notes will be added throughout development.*
