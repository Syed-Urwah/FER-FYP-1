"use client"
import { useSession, signOut } from 'next-auth/react';
import { Button } from '@/components/ui/button';
import { Card, CardHeader, CardTitle } from '@/components/ui/card';

export default function Header() {
    const { data: session, status } = useSession();
    const loading = status === 'loading';

    return (
        <Card className="w-full max-w-6xl mx-auto my-4">
            <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle className="text-lg">
                    {loading ? 'Loading...' : session?.user?.name ? `Welcome, ${session.user.name}` : 'Not signed in'}
                </CardTitle>
                {session?.user?.email && (
                    <Button variant="outline" onClick={() => signOut({ callbackUrl: '/' })}>
                        Logout
                    </Button>
                )}
            </CardHeader>
        </Card>
    );
}
