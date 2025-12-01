import * as React from 'react';

// Simple Table components using Tailwind CSS, compatible with shadcn/ui styling.

export function Table({ children, className }: { children: React.ReactNode; className?: string }) {
    return (
        <div className={`rounded-md border ${className || ''}`}>
            <table className="w-full caption-bottom text-sm">
                {children}
            </table>
        </div>
    );
}

export function TableHeader({ children }: { children: React.ReactNode }) {
    return <thead className="border-b bg-muted/50">{children}</thead>;
}

export function TableBody({ children }: { children: React.ReactNode }) {
    return <tbody>{children}</tbody>;
}

export function TableRow({ children, className }: { children: React.ReactNode; className?: string }) {
    return <tr className={className}>{children}</tr>;
}

export function TableHead({ children, className }: { children: React.ReactNode; className?: string }) {
    return (
        <th
            className={`h-12 px-4 text-left align-middle font-medium text-muted-foreground ${className || ''}`}
        >
            {children}
        </th>
    );
}

export function TableCell({ children, className }: { children: React.ReactNode; className?: string }) {
    return (
        <td className={`p-4 align-middle ${className || ''}`}>{children}</td>
    );
}
