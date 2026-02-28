import React, { useState, useMemo } from 'react';
import type { FishInstance } from '../types/annotations';
import { getSpeciesLabel, getSpeciesColor } from '../lib/speciesColors';

interface FishTableProps {
  instances: FishInstance[];
  selectedTrackId: number | null;
  onSelectTrack: (trackId: number) => void;
}

type SortKey = 'trackId' | 'speciesId' | 'length_cm' | 'weight_g_est' | 'confidence';

/** Sortable table listing all fish tracks in the current frame. */
const FishTable: React.FC<FishTableProps> = ({ instances, selectedTrackId, onSelectTrack }) => {
  const [sortKey, setSortKey] = useState<SortKey>('trackId');
  const [sortAsc, setSortAsc] = useState(true);

  const sorted = useMemo(() => {
    const copy = [...instances];
    copy.sort((a, b) => {
      let av: number | string;
      let bv: number | string;
      switch (sortKey) {
        case 'trackId':
          av = a.trackId; bv = b.trackId; break;
        case 'speciesId':
          av = a.speciesId; bv = b.speciesId; break;
        case 'length_cm':
          av = a.metrics.length_cm; bv = b.metrics.length_cm; break;
        case 'weight_g_est':
          av = a.metrics.weight_g_est; bv = b.metrics.weight_g_est; break;
        case 'confidence':
          av = a.metrics.confidence; bv = b.metrics.confidence; break;
        default:
          av = a.trackId; bv = b.trackId;
      }
      if (av < bv) return sortAsc ? -1 : 1;
      if (av > bv) return sortAsc ? 1 : -1;
      return 0;
    });
    return copy;
  }, [instances, sortKey, sortAsc]);

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortAsc(!sortAsc);
    } else {
      setSortKey(key);
      setSortAsc(true);
    }
  };

  const sortIndicator = (key: SortKey) => {
    if (sortKey !== key) return '';
    return sortAsc ? ' ▲' : ' ▼';
  };

  return (
    <div className="tracks-table-wrapper">
      <h3 className="panel-heading">Fish Tracks</h3>
      <div className="tracks-table-scroll">
        <table className="tracks-table">
          <colgroup>
            <col className="tracks-col-id" />
            <col className="tracks-col-species" />
            <col className="tracks-col-length" />
            <col className="tracks-col-weight" />
            <col className="tracks-col-confidence" />
          </colgroup>
          <thead>
            <tr>
              <th className="tracks-th tracks-th-id" onClick={() => handleSort('trackId')}>ID{sortIndicator('trackId')}</th>
              <th className="tracks-th tracks-th-species" onClick={() => handleSort('speciesId')}>Species{sortIndicator('speciesId')}</th>
              <th className="tracks-th tracks-th-length" onClick={() => handleSort('length_cm')}>Length{sortIndicator('length_cm')}</th>
              <th className="tracks-th tracks-th-weight" onClick={() => handleSort('weight_g_est')}>Weight{sortIndicator('weight_g_est')}</th>
              <th className="tracks-th tracks-th-confidence" onClick={() => handleSort('confidence')}>Confidence{sortIndicator('confidence')}</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((fish) => (
              <tr
                key={fish.trackId}
                className={`tracks-row ${fish.trackId === selectedTrackId ? 'tracks-row-selected' : ''}`}
                onClick={() => onSelectTrack(fish.trackId)}
              >
                <td className="tracks-td tracks-td-id">{fish.trackId}</td>
                <td className="tracks-td tracks-td-species">
                  <span className="tracks-species-content">
                    <span
                      className="tracks-species-dot"
                      style={{ backgroundColor: getSpeciesColor(fish.speciesId) }}
                    />
                    {getSpeciesLabel(fish.speciesId)}
                  </span>
                </td>
                <td className="tracks-td tracks-td-length">{fish.metrics.length_cm.toFixed(1)}</td>
                <td className="tracks-td tracks-td-weight">{fish.metrics.weight_g_est.toFixed(0)}</td>
                <td className="tracks-td tracks-td-confidence">{(fish.metrics.confidence * 100).toFixed(0)}%</td>
              </tr>
            ))}
            {sorted.length === 0 && (
              <tr>
                <td colSpan={5} className="tracks-table-empty">No fish in current frame</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default FishTable;
