import Pivot, { PivotProps } from 'hew/Pivot';
import { Loaded } from 'hew/utils/loadable';
import { default as MarkdownViewer } from 'markdown-to-jsx';
import React, { useMemo } from 'react';

import CodeEditor from 'components/CodeEditor';
import useResize from 'hooks/useResize';
import handleError from 'utils/error';

import css from './Markdown.module.scss';

interface Props {
  disabled?: boolean;
  editing?: boolean;
  markdown: string;
  onChange?: (editedMarkdown: string) => void;
  onClick?: (e: React.MouseEvent) => void;
}

interface RenderProps {
  markdown: string;
  onClick?: (e: React.MouseEvent) => void;
  placeholder?: string;
}

const TabType = {
  Edit: 'edit',
  Preview: 'preview',
} as const;

const MarkdownRender: React.FC<RenderProps> = ({ markdown, placeholder, onClick }) => {
  const showPlaceholder = !markdown && placeholder;
  return (
    <div className={css.render} onClick={onClick}>
      {showPlaceholder ? (
        <div className={css.placeholder}>{placeholder}</div>
      ) : (
        <MarkdownViewer options={{ disableParsingRawHTML: true }}>{markdown}</MarkdownViewer>
      )}
    </div>
  );
};

const Markdown: React.FC<Props> = ({
  disabled = false,
  editing = false,
  markdown,
  onChange,
  onClick,
}: Props) => {
  const resize = useResize();
  const tabItems: PivotProps['items'] = useMemo(() => {
    return [
      {
        children: (
          <div className={css.noOverflow}>
            <CodeEditor
              file={Loaded(markdown)}
              files={[{ key: 'input.md' }]}
              height={`${resize.height - 420}px`}
              onChange={onChange}
              onError={handleError}
            />
          </div>
        ),
        key: TabType.Edit,
        label: 'Edit',
      },
      {
        children: <MarkdownRender markdown={markdown} onClick={onClick} />,
        key: TabType.Preview,
        label: 'Preview',
      },
    ];
  }, [markdown, onChange, onClick, resize]);

  return (
    <div aria-label="markdown-editor" className={css.base} tabIndex={0}>
      {editing && !disabled ? (
        <Pivot items={tabItems} />
      ) : (
        <MarkdownRender
          markdown={markdown}
          placeholder={disabled ? 'No note present.' : 'Add notes...'}
          onClick={onClick}
        />
      )}
    </div>
  );
};

export default Markdown;
