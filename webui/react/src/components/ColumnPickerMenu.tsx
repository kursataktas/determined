import Badge from 'hew/Badge';
import Button from 'hew/Button';
import Checkbox, { CheckboxChangeEvent } from 'hew/Checkbox';
import Dropdown from 'hew/Dropdown';
import Icon from 'hew/Icon';
import Input from 'hew/Input';
import Message from 'hew/Message';
import Pivot from 'hew/Pivot';
import Spinner from 'hew/Spinner';
import { Loadable } from 'hew/utils/loadable';
import React, { ChangeEvent, useCallback, useMemo, useState } from 'react';
import { FixedSizeList as List } from 'react-window';

import { V1LocationType } from 'services/api-ts-sdk';
import { ProjectColumn } from 'types';
import { ensureArray } from 'utils/data';

import css from './ColumnPickerMenu.module.scss';

const BANNED_COLUMNS: Set<string> = new Set([]);

const removeBannedColumns = (columns: ProjectColumn[]) =>
  columns.filter((col) => !BANNED_COLUMNS.has(col.column));

export const LOCATION_LABEL_MAP: Record<V1LocationType, string> = {
  [V1LocationType.EXPERIMENT]: 'General',
  [V1LocationType.RUN]: 'General',
  [V1LocationType.VALIDATIONS]: 'Metrics',
  [V1LocationType.TRAINING]: 'Metrics',
  [V1LocationType.CUSTOMMETRIC]: 'Metrics',
  [V1LocationType.HYPERPARAMETERS]: 'Hyperparameters',
  [V1LocationType.RUNHYPERPARAMETERS]: 'Hyperparameters',
  [V1LocationType.RUNMETADATA]: 'Metadata',
  [V1LocationType.UNSPECIFIED]: 'Unspecified',
} as const;

export const COLUMNS_MENU_BUTTON = 'columns-menu-button';

interface ColumnMenuProps {
  isMobile?: boolean;
  initialVisibleColumns: [string, string][];
  defaultVisibleColumns: [string, string][];
  defaultPinnedCount: number;
  onVisibleColumnChange?: (newColumns: string[], pinnedCount?: number) => void;
  onHeatmapSelectionRemove?: (id: string) => void;
  projectColumns: Loadable<ProjectColumn[]>;
  projectId: number;
  tabs: (V1LocationType | V1LocationType[])[];
  compare?: boolean;
  pinnedColumnsCount: number;
}

interface ColumnTabProps {
  columnState: [string, string][];
  handleShowSuggested: () => void;
  onVisibleColumnChange?: (newColumns: string[], pinnedCount?: number) => void;
  projectId: number;
  searchString: string;
  setSearchString: React.Dispatch<React.SetStateAction<string>>;
  tab: V1LocationType | V1LocationType[];
  totalColumns: ProjectColumn[];
  compare: boolean;
  pinnedColumnsCount: number;
  onHeatmapSelectionRemove?: (id: string) => void;
}

const ColumnPickerTab: React.FC<ColumnTabProps> = ({
  columnState,
  compare,
  pinnedColumnsCount,
  handleShowSuggested,
  searchString,
  setSearchString,
  tab,
  totalColumns,
  onVisibleColumnChange,
  onHeatmapSelectionRemove,
}) => {
  const checkedColumnNames = useMemo(
    () => (compare ? columnState.slice(0, pinnedColumnsCount) : columnState),
    [columnState, compare, pinnedColumnsCount],
  );

  const filteredColumns = useMemo(() => {
    const regex = new RegExp(searchString, 'i');
    const locations = ensureArray(tab);
    return totalColumns
      .filter(
        (col) => locations.includes(col.location) && regex.test(col.displayName || col.column),
      )
      .sort(
        (a, b) =>
          locations.findIndex((l) => l === a.location) -
          locations.findIndex((l) => l === b.location),
      );
  }, [searchString, totalColumns, tab]);

  const allFilteredColumnsChecked = useMemo(() => {
    return filteredColumns.every((col) => {
      const found = columnState.find(
        ([type, column]) => type === col.type && column === col.column,
      );

      return found !== undefined;
    });
  }, [columnState, filteredColumns]);

  const handleShowHideAll = useCallback(() => {
    const filteredColumnMap: Record<string, boolean> = filteredColumns.reduce((acc, col) => {
      const found =
        columnState.find(([type, column]) => type === col.type && column === col.column) !==
        undefined;

      return {
        ...acc,
        [col.type.concat(`/${col.column}`)]: found,
      };
    }, {});

    const newColumns = allFilteredColumnsChecked
      ? columnState.filter(([type, col]) => !filteredColumnMap[type.concat(`/${col}`)])
      : [...columnState, ...filteredColumns.map<[string, string]>((col) => [col.type, col.column])]; // TODO: check if that needs to be mapped with the metadata
    const pinnedCount = allFilteredColumnsChecked
      ? // If uncheck something pinned, reduce the pinnedColumnsCount
        newColumns.filter(([type, col]) => {
          let stateIndex: number = -1;
          for (const [index, [stateColumnType, stateColumn]] of columnState.entries()) {
            if (col === stateColumn && type === stateColumnType) {
              stateIndex = index;
              break;
            }
          }

          return stateIndex < pinnedColumnsCount;
        }).length
      : pinnedColumnsCount;

    onVisibleColumnChange?.(
      newColumns.map(([type, col]) => type.concat(`/${col}`)),
      pinnedCount,
    );
  }, [
    allFilteredColumnsChecked,
    columnState,
    filteredColumns,
    onVisibleColumnChange,
    pinnedColumnsCount,
  ]);

  const handleColumnChange = useCallback(
    (event: CheckboxChangeEvent) => {
      const { id, checked } = event.target;

      if (id === undefined) return;

      const [type, targetCol] = id.split('/');

      if (compare) {
        // pin or unpin column
        const newColumns = columnState.filter(([t, c]) => c !== targetCol && t !== type);
        let pinnedCount = pinnedColumnsCount;
        if (checked) {
          newColumns.splice(pinnedColumnsCount, 0, [type, targetCol]);
          pinnedCount = Math.max(pinnedColumnsCount + 1, 0);
        } else {
          newColumns.splice(pinnedColumnsCount - 1, 0, [type, targetCol]);
          pinnedCount = Math.max(pinnedColumnsCount - 1, 0);
        }
        onVisibleColumnChange?.(
          newColumns.map(([type, col]) => type.concat(`/${col}`)),
          pinnedCount,
        );
      } else {
        let pinnedCount = pinnedColumnsCount;
        let stateIndex: number = -1;
        for (const [index, [stateColumn, stateColumnType]] of columnState.entries()) {
          if (targetCol === stateColumn && type === stateColumnType) {
            stateIndex = index;
            break;
          }
        }
        // If uncheck something pinned, reduce the pinnedColumnsCount
        if (!checked && stateIndex < pinnedColumnsCount) {
          pinnedCount = Math.max(pinnedColumnsCount - 1, 0);
        }
        // If uncheck something had heatmap skipped, reset to heatmap visible
        if (!checked) {
          onHeatmapSelectionRemove?.(targetCol);
        }
        const newColumnSet = [...columnState];
        checked ? newColumnSet.push([type, targetCol]) : newColumnSet.splice(stateIndex, 1);
        onVisibleColumnChange?.(
          [...newColumnSet.map(([type, col]) => type.concat(`/${col}`))],
          pinnedCount,
        );
      }
    },
    [compare, columnState, onVisibleColumnChange, onHeatmapSelectionRemove, pinnedColumnsCount],
  );

  const handleSearch = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      setSearchString(e.target.value);
    },
    [setSearchString],
  );

  const rows = useCallback(
    ({ index, style }: { index: number; style: React.CSSProperties }) => {
      const col = filteredColumns[index];
      const getColDisplayName = () => (
        <>
          {col.displayName || col.column} <Badge text={col.type.replace('COLUMN_TYPE_', '')} />
        </>
      );
      const getChecked = () => {
        const checked =
          checkedColumnNames.find(
            ([type, column]) => type === col.type && column === col.column,
          ) !== undefined;
        return checked;
      };
      return (
        <div
          className={css.rows}
          data-test="row"
          data-test-id={`${col.type}/${col.column}`}
          key={`${col.type}/${col.column}`}
          style={style}>
          <Checkbox
            checked={getChecked()}
            data-test="checkbox"
            id={`${col.type}/${col.column}`}
            onChange={handleColumnChange}>
            {getColDisplayName()}
          </Checkbox>
        </div>
      );
    },
    [filteredColumns, checkedColumnNames, handleColumnChange],
  );

  return (
    <div data-test-component="columnPickerTab" data-testid="column-picker-tab">
      <Input
        allowClear
        autoFocus
        data-test="search"
        placeholder="Search"
        value={searchString}
        onChange={handleSearch}
      />
      {totalColumns.length !== 0 ? (
        <div className={css.columns} data-test="columns">
          {filteredColumns.length > 0 ? (
            <List height={360} itemCount={filteredColumns.length} itemSize={30} width="100%">
              {rows}
            </List>
          ) : (
            <Message description="No results" icon="warning" />
          )}
        </div>
      ) : (
        <Spinner spinning />
      )}
      {!compare && (
        <div className={css.actionRow}>
          <Button data-test="showAll" type="text" onClick={handleShowHideAll}>
            {allFilteredColumnsChecked ? 'Hide' : 'Show'} all
          </Button>
          <Button data-test="reset" type="text" onClick={handleShowSuggested}>
            Reset
          </Button>
        </div>
      )}
    </div>
  );
};

const ColumnPickerMenu: React.FC<ColumnMenuProps> = ({
  compare = false,
  pinnedColumnsCount,
  projectColumns,
  initialVisibleColumns,
  defaultVisibleColumns,
  defaultPinnedCount,
  projectId,
  isMobile = false,
  onVisibleColumnChange,
  onHeatmapSelectionRemove,
  tabs,
}) => {
  const [searchString, setSearchString] = useState('');
  const [open, setOpen] = useState(false);

  const closeMenu = () => {
    setOpen(false);
  };

  const handleOpenChange = (newOpen: boolean) => {
    setOpen(newOpen);
  };

  const totalColumns = useMemo(
    () => removeBannedColumns(Loadable.getOrElse([], projectColumns)),
    [projectColumns],
  );

  const handleShowSuggested = useCallback(() => {
    onVisibleColumnChange?.(
      defaultVisibleColumns.map(([type, col]) => type.concat(`/${col}`)),
      defaultPinnedCount,
    );
    closeMenu();
  }, [onVisibleColumnChange, defaultVisibleColumns, defaultPinnedCount]);

  return (
    <Dropdown
      content={
        <div className={css.base}>
          {tabs.length > 1 && (
            <Pivot
              items={tabs.map((tab) => {
                const canonicalTab = Array.isArray(tab) ? tab[0] : tab;
                return {
                  children: (
                    <ColumnPickerTab
                      columnState={initialVisibleColumns}
                      compare={compare}
                      handleShowSuggested={handleShowSuggested}
                      pinnedColumnsCount={pinnedColumnsCount}
                      projectId={projectId}
                      searchString={searchString}
                      setSearchString={setSearchString}
                      tab={tab}
                      totalColumns={totalColumns}
                      onHeatmapSelectionRemove={onHeatmapSelectionRemove}
                      onVisibleColumnChange={onVisibleColumnChange}
                    />
                  ),
                  forceRender: true,
                  key: canonicalTab,
                  label: LOCATION_LABEL_MAP[canonicalTab],
                };
              })}
            />
          )}
          {tabs.length === 1 && (
            <ColumnPickerTab
              columnState={initialVisibleColumns}
              compare={compare}
              handleShowSuggested={handleShowSuggested}
              pinnedColumnsCount={pinnedColumnsCount}
              projectId={projectId}
              searchString={searchString}
              setSearchString={setSearchString}
              tab={tabs[0]}
              totalColumns={totalColumns}
              onVisibleColumnChange={onVisibleColumnChange}
            />
          )}
        </div>
      }
      open={open}
      onOpenChange={handleOpenChange}>
      <Button
        data-test-component="columnPickerMenu"
        data-testid={COLUMNS_MENU_BUTTON}
        hideChildren={isMobile}
        icon={<Icon name="columns" title="column picker" />}>
        Columns
      </Button>
    </Dropdown>
  );
};

export default ColumnPickerMenu;
