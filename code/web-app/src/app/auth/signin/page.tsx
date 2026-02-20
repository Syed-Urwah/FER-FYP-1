'use client';

import { signIn } from 'next-auth/react';
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { Spinner } from '@/components/ui/spinner';

export default function SignIn() {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [isLoading, setIsLoading] = useState(false)
    const router = useRouter();

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setIsLoading(true)
        const result = await signIn('credentials', {
            email: username, // NextAuth expects 'email' based on our config, but UI says 'Username' (which is actually email in our register flow)
            password,
            redirect: false,
        });

        setIsLoading(false)
        if (result?.ok) {
            router.push('/');
        } else {
            alert('Invalid credentials');
        }
    };

    return (
        <div className="flex min-h-screen items-center justify-center bg-slate-50">
            <Card className="w-full max-w-md">
                <CardHeader>
                    <CardTitle className="text-center">Sign In</CardTitle>
                </CardHeader>
                <CardContent>
                    <form onSubmit={handleSubmit} className="space-y-4">
                        <div>
                            <label className="block text-sm font-medium text-slate-700">Email</label>
                            <Input
                                type="email"
                                value={username}
                                onChange={(e) => setUsername(e.target.value)}
                                placeholder="john@example.com"
                                required
                            />
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-slate-700">Password</label>
                            <Input
                                type="password"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                placeholder="••••••••"
                                required
                            />
                        </div>
                        <Button disabled={isLoading} type="submit" className="w-full">
                            {isLoading ? <Spinner/> : "Sign In"}
                        </Button>
                        <div className="text-center text-sm text-slate-500">
                            Don't have an account? <Link href="/auth/register" className="text-indigo-600 hover:underline">Register</Link>
                        </div>
                    </form>
                </CardContent>
            </Card>
        </div>
    );
}
